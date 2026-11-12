# Local research database

PostgreSQL 18 runs on a group machine, reachable from approved devices on the
group's private network and locally as `localhost:5432`. See
[docs/setup.md](../docs/setup.md) for getting access.

| Folder or file | What |
|---|---|
| `schema.sql` (`tables.sql`, `constraints.sql`) | builds the current schema on an empty database |
| `migrations/` | one file per schema change since the migration from Supabase |
| `roles.sql`, `agent_schema_access.sql`, `agent_guards.sql` | roles and the AI-agent guards, applied once by an administrator |
| `queries/` | saved export queries; run with `research export --sql-file` |
| `loads/` | how the tables filled from CSV files were loaded |
| `fixes/` | one-off data changes, as run |
| `verify.sql`, `backup.ps1` | post-restore checks and the nightly backup |

## Schema and migration

`schema.sql` applies `tables.sql` then `constraints.sql` to an empty database.
For the large Supabase copy, apply `tables.sql`, restore the data with
`pg_restore --data-only`, then apply `constraints.sql` and `roles.sql`.
The indexes and foreign keys are built and checked after the copy.
`roles.sql` expects the `research_app`, `research_ro` and `research_agent`
login roles to exist already; an administrator creates them and sets their
passwords. On a new database, then apply `agent_schema_access.sql` and
`agent_guards.sql` as `postgres` (see "Agent access" below).

Schema changes made after the migration go in `migrations/`, one file per
change named `YYYYMMDD_HHMM_<what>.sql`. Each change is also folded into
`tables.sql` or `constraints.sql`, so `schema.sql` still builds the current
schema from scratch.

## Roles

| Role | Used by | Access |
|---|---|---|
| `postgres` | local administration only | superuser; never allowed over the network |
| `research_app` | the upload pipeline | read and write rows; no schema changes |
| `research_ro` | collaborators doing analysis, exports | read only |
| `research_agent` | an AI coding agent | read and write rows, owns the `public` tables and can change the schema; drops and truncates are refused unless approved |

## Application connection

Use the `research_app` role for uploads. Set `DATABASE_URL` in the process
environment to `postgresql://research_app@localhost:5432/research` on the
desktop or `postgresql://research_app@<windows-tailscale-hostname>:5432/research`
on another Tailscale device. Store the password in that machine's PostgreSQL
password file rather than in the URL. On Windows, this file is
`%APPDATA%\postgresql\pgpass.conf`; on macOS, it is `~/.pgpass`.

`research ingest` writes companies and documents (`research.db.documents`), and
`research backfill` writes exposure runs and keyword matches (`research.db.exposures`).
The `research_ro` account can read the database but cannot upload or change rows.

## Changing data

Every change to rows or schema is a SQL file in the repo, so the database can be rebuilt:
a schema change goes in `migrations/` (and is folded into `tables.sql` or
`constraints.sql`), a one-off fix in `fixes/` with a guard that makes it a no-op if the
rows drifted, and a table filled from a CSV gets a script in `loads/`. Back up the rows a
fix changes under `data/private/db_fixes/` first.

## Tests

`pytest -m db` runs the integration tests against `TEST_DATABASE_URL`, which must be a
throwaway database built with `schema.sql` (the tests refuse a database named
`research`). A local Postgres works:
`createdb scratch_test && psql -d scratch_test -f db/schema.sql`.

## Agent access

`research_agent` owns the `public` tables and their identity sequences and has
`CREATE` on the schema, so it can add tables, columns, indexes and constraints
(`agent_schema_access.sql`). Default privileges give `research_app` read and
write access and `research_ro` read access to every table it creates.

`agent_guards.sql` installs guards in a superuser-owned `admin` schema. They
apply only to sessions logged in as `research_agent`:

- An event trigger refuses any drop: tables, columns, constraints, indexes,
  defaults, triggers and so on.
- A `BEFORE TRUNCATE` trigger on every table refuses `TRUNCATE`. New tables
  get it automatically, and attempts to disable it are refused.

A drop or truncate the user has approved runs in a transaction that first sets
`research.allow_drop = on`, for example:

```sql
DO $$
BEGIN
    PERFORM set_config('research.allow_drop', 'on', true);
    DROP TABLE public.some_table;
END
$$;
```

The flag lasts only until the end of that transaction. Getting the user's
approval happens in the agent's tooling, which must ask before running any
statement that sets the flag. The database guard makes sure an unapproved
drop fails even if a statement gets past that check. `DELETE` and permission
changes are not blocked in the database, so the agent's tooling must ask
before those as well.

For Claude Code that tooling is `.claude/hooks/db_write_guard.py`, which the shared
`.claude/settings.json` runs before every research-db write tool call. Routine writes
(inserts, updates with a WHERE, new tables and columns) run without a prompt; drops,
truncates, deletes, updates without a WHERE, column type changes, permission changes,
bulk CSV loads and COPY, EXECUTE, CALL and anything that can run code later need the
user's approval. It logs each decision
to `db_write_guard.log` beside it (git-ignored). The command calls `/usr/bin/python3`,
which exists on macOS and Linux; on Windows, override it in `.claude/settings.local.json`.

Tables that `postgres` creates stay owned by `postgres`, and the agent can't
alter them. To hand one over, run
`ALTER TABLE public.<table> OWNER TO research_agent;`.

## Backups

`backup.ps1` makes two kinds of compressed custom-format dump of `research`:

- `research_<stamp>.dump`, every night: everything except the rows of
  `transcript_files`. Successful dumps are kept for 7 days.
- `transcript_files_<stamp>.dump`, only when `transcript_files` has gained rows
  since the last one. The table is append-only and holds the raw vendor XMLs, so
  its rows are copied once per load, not once per night. One copy is kept, and
  `transcript_files.state` records the row count it was made at.

Its default destination is `G:\pg_backups`;
pass `-BackupDir` to select another location. The script uses a restricted
password file at `G:\pg_backups\pgpass.conf` containing only the local
`postgres` login entry. If that password changes, update this file too.
It logs success or failure to `backup.log`.

### Restoring

A nightly dump alone restores an empty `transcript_files`, so a restore takes both
dumps, in this order, into a new empty database:

```
pg_restore -U postgres -d <new database> research_<stamp>.dump
pg_restore -U postgres -d <new database> --data-only transcript_files_<stamp>.dump
```

Nothing has a foreign key into `transcript_files` (other tables name a revision by its
SHA-256), so the order only matters for having the table to load into. If the
transcript dump is lost, rebuild the rows from the vendor zips with
`research transcripts load`; the hashes come out the same, the `id` values may not.
Afterwards run `research transcripts verify` against the zips.

Both steps were tested on 2026-11-11 into a scratch database (without the rows of
`keyword_matches`, to keep it short): all 635,490 transcript rows came back, every one
still hashed to its stored SHA-256, the `id` sequence continued from the highest id, and
the append-only grants were intact. The transcript dump was 10.4 GB and took six minutes
to make and five to restore.

Windows Task Scheduler runs the stable copy at `G:\pg_backups\backup.ps1`
daily while the Windows user is signed in. The archive
and login-file folder grants access only to that user, SYSTEM, and local
administrators. The first backup was restored and its five table counts,
identity sequences, and foreign keys matched the original database.

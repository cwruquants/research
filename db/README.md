# Local research database

PostgreSQL 18 runs on a Windows desktop. It is reachable from approved
Tailscale devices using that desktop's Tailscale hostname and locally as
`localhost:5432`. The source Supabase project remains a separate fallback.

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

The uploader is `PostgresUploader` in `src/analysis/postgres_uploader.py`, or
call `BatchRunner.process_batch_to_postgres(...)`. The Supabase uploader is
still available. The `research_ro` account can read the database but cannot
upload or change rows.

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

Tables that `postgres` creates stay owned by `postgres`, and the agent can't
alter them. To hand one over, run
`ALTER TABLE public.<table> OWNER TO research_agent;`.

## Backups

`backup.ps1` makes a compressed custom-format dump of `research` and keeps
successful dumps for 14 days. Its default destination is `G:\pg_backups`;
pass `-BackupDir` to select another location. The script uses a restricted
password file at `G:\pg_backups\pgpass.conf` containing only the local
`postgres` login entry. If that password changes, update this file too.
It logs success or failure to `backup.log`.

Windows Task Scheduler runs the stable copy at `G:\pg_backups\backup.ps1`
daily while the Windows user is signed in. The archive
and login-file folder grants access only to that user, SYSTEM, and local
administrators. The first backup was restored and its five table counts,
identity sequences, and foreign keys matched the original database.

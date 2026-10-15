# Local research database

PostgreSQL 18 runs on the Windows desktop. It is reachable from the tailnet as
`desktop-64ns9eu:5432` and locally as `localhost:5432`. The source Supabase
project remains a separate fallback.

## Schema and migration

`schema.sql` applies `tables.sql` then `constraints.sql` to an empty database.
For the large Supabase copy, apply `tables.sql`, restore the data with
`pg_restore --data-only`, then apply `constraints.sql` and `roles.sql`.
The indexes and foreign keys are built and checked after the copy.

## Application connection

Use the `research_app` role for uploads. Set `DATABASE_URL` in the process
environment to `postgresql://research_app@localhost:5432/research` on the
desktop or `postgresql://research_app@desktop-64ns9eu:5432/research` on the
Mac. Store the password in that machine's PostgreSQL password file rather
than in the URL. On Windows, this file is
`%APPDATA%\postgresql\pgpass.conf`; on macOS, it is `~/.pgpass`.

The uploader is `PostgresUploader` in `src/analysis/postgres_uploader.py`, or
call `BatchRunner.process_batch_to_postgres(...)`. The Supabase uploader is
still available. The `research_ro` account can read the database but cannot
upload or change rows.

## Backups

`backup.ps1` makes a compressed custom-format dump of `research` and keeps
successful dumps for 14 days. Its default destination is `G:\pg_backups`;
pass `-BackupDir` to select another location. The script uses a restricted
password file at `G:\pg_backups\pgpass.conf` containing only the local
`postgres` login entry. If that password changes, update this file too.
It logs success or failure to `backup.log`.

Windows Task Scheduler runs the stable copy at `G:\pg_backups\backup.ps1`
daily at 3:00 a.m. while the `arete` Windows user is signed in. The archive
and login-file folder grants access only to that user, SYSTEM, and local
administrators. The first backup was restored and its five table counts,
identity sequences, and foreign keys matched the original database.

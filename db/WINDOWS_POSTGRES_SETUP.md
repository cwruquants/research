# Handoff: self-hosted Postgres on the Windows desktop

You are taking over a task that was planned on the user's Mac. Your job is to turn the user's Windows desktop into the project's database server: install PostgreSQL 18, make it reachable **only over Tailscale**, copy the existing data out of Supabase, add a direct-Postgres uploader to the Python code, and set up backups.

Work through the phases in order. Each phase ends with a **completion check**: do not start the next phase until it passes. Steps marked **GATE** need the user's explicit yes in chat before you run them. When something in this doc disagrees with what you observe on the machine, trust the machine, tell the user, and adapt.

---

## 1. Context

### The project
- Repo: `cwruquants/research` on GitHub (`git@github.com:cwruquants/research.git`). This is a **shared organization repo**, so anything you push is visible to the user's collaborators.
- It's a Python research pipeline that analyzes earnings call transcripts (sentiment, readability, keyword exposure) and uploads the results.
- Today, results go to a hosted **Supabase** project (free tier) through `src/analysis/supabase_uploader.py` (`SupabaseUploader`), called from `BatchRunner.process_batch_to_supabase()` in `src/analysis/batch_runner.py`. The caller builds the Supabase client and passes it in; nothing in the repo creates it.
- Python env: `environment.yml` defines a conda env named `research` (Python 3.10). Check whether it exists on this machine (`conda env list`) before relying on it.
- `requirements.txt` is **UTF-16 LE with BOM and CRLF line endings** (it came from `pip freeze` in PowerShell). Keep that encoding when you edit it, or the file will end up with mixed encodings. In PowerShell: `Add-Content -Encoding Unicode`. pip reads UTF-16 with a BOM fine.

### Why the move
The user wants to keep everything without paying for a subscription. The Supabase free tier caps the database at 500 MB and pauses after 7 idle days. The desktop has plenty of room and is otherwise mostly idle.

### Machines (Tailscale tailnet, owner account `aretelew@gmail.com`)
| Machine | Tailscale name | Tailscale IPv4 | Role |
|---|---|---|---|
| Windows desktop (you are here) | `desktop-64ns9eu` | `100.81.99.98` | Database server |
| MacBook Pro | `anthonys-macbook-pro` | `100.96.143.81` | User's laptop, remote client |

Already verified: on school Wi-Fi, `tailscale netcheck` shows `UDP: true`, and Mac → desktop pings show a **direct connection at about 8 ms**. Windows username is `arete` (`C:\Users\arete`).

### Supabase source database
| Item | Value |
|---|---|
| Project ref | `augusbsnxlezrehaksfx` (project name `data`) |
| Postgres version | 17.6 |
| Connection to use | **Session pooler**: host `aws-0-us-west-2.pooler.supabase.com`, port `5432`, db `postgres`, user `postgres.augusbsnxlezrehaksfx`, `sslmode=require` |
| Password | The user's Supabase **database** password (see the secrets rules in section 3) |

Use the session pooler, not the direct host `db.<ref>.supabase.co`, because the direct host is IPv6-only on the free plan. Don't use port 6543 either: that's transaction mode and pg_dump breaks on it.

The schema below comes from the migration file `supabase/migrations/20260904170135_remote_schema.sql` on the Mac. It is **not** in git, so you won't have it. The live database may have changed since September 4, 2026, and Phase 2 checks for that.

---

## 2. Decisions already made (don't reopen these)

| Decision | Why |
|---|---|
| **PostgreSQL 18** (latest 18.x), EDB installer | It's the current stable release. Its pg_dump can read Supabase's 17.6 server. PG 19 is only at RC1 (released 2026-10-15), so it's too new. |
| **No pgvector** | The Supabase schema enables `vector`, but no table uses a vector column. Embeddings are computed in memory in `matching_agent.py`. Skipping it also avoids building pgvector with Visual Studio. |
| **Drop Supabase-only pieces** | Drop the `supabase_vault`, `wrappers`, `pg_graphql` and `pg_stat_statements` extensions, all RLS policies and `ENABLE ROW LEVEL SECURITY`, the anon/authenticated/service_role grants, the `supabase_realtime` publication, and the `public.execute_sql(text)` function (a `SECURITY DEFINER` function that runs any SQL: a security hole). `pgcrypto` and `uuid-ossp` aren't used by any column either, so skip them. |
| **Reachable only via Tailscale** | Three layers: pg_hba allows only Tailscale address ranges, Windows Firewall allows only Tailscale ranges on the Tailscale adapter, and nothing is port-forwarded or exposed publicly. No TLS on Postgres, because WireGuard already encrypts the traffic. |
| **`listen_addresses = '*'`**, not the 100.x IP | If Postgres starts before the Tailscale adapter is up, binding to the 100.x IP silently fails and stays failed until Postgres restarts. Access is restricted by pg_hba and the firewall instead. |
| **Low resource use** | The user games on this machine and wants Postgres to stay out of the way. The database is under 500 MB, so a 1 GB cache holds all of it. Don't tune for throughput. |
| **Roles** | `postgres` is the superuser, local admin only, never allowed remotely. `research_app` reads and writes (the user and analysis machines). `research_ro` is read-only (future collaborators). |
| **Keep `SupabaseUploader` untouched** | The user explicitly wants a **separate** `PostgresUploader` copy next to it, so both paths keep working. |
| **Supabase stays as a fallback** | Copy the data; never delete or modify anything in Supabase. |
| **Fix duplicate NULL-ticker companies** | `UNIQUE (name, ticker)` treats NULLs as distinct, so companies with no ticker can be inserted over and over. The new schema uses `UNIQUE NULLS NOT DISTINCT` (available since PG 15). |

---

## 3. Ground rules

### Secrets
- **The user types every password. You never see, generate, print or store one.**
  - Passwords for Supabase and the local superuser go into `%APPDATA%\postgresql\pgpass.conf` (`C:\Users\arete\AppData\Roaming\postgresql\pgpass.conf`). The user edits it in Notepad. libpq tools (psql, pg_dump, pg_restore) read it automatically. Line format: `host:port:database:username:password`. A `:` or `\` inside a password must be escaped as `\:` or `\\`.
  - Passwords for new roles are set by the user in an interactive `psql` session with `\password <role>`.
- Never put a password in a command line, a script, a committed file, a URL query string, or your output. `.env` is gitignored and is where the user stores `DATABASE_URL` for the Python code.
- If a command fails on authentication, ask the user to check the pgpass entry. Don't ask them to paste the password to you.

### Admin rights and system changes
Firewall rules, power settings and services need an **elevated** PowerShell. If your shell isn't elevated, or your own rules don't allow you to change system or security settings, write out the exact commands and have the user run them in "Terminal (Admin)", then check the result yourself.

### GATE list (ask first, then wait for a clear yes)
- Every firewall change, including disabling any existing rule.
- Any power, sleep or Windows Update setting.
- Changing data while deduplicating companies (Phase 5).
- Deleting anything (files, rows, roles, databases).
- `git push`, and opening a PR on the shared repo.

### Reporting
After each phase, tell the user in 2–4 lines what changed and whether the completion check passed.

---

## 4. Phases

### Phase 1: Install PostgreSQL 18

1. Check whether Postgres is already installed: `Get-Service postgresql*`, and look for `C:\Program Files\PostgreSQL\`. If a version is installed, stop and ask the user how to proceed.
2. Install with the GUI so the user can type the superuser password:
   ```powershell
   winget install -e --id PostgreSQL.PostgreSQL.18 --interactive
   ```
   If winget isn't available, the user downloads the EDB installer from https://www.postgresql.org/download/windows/.
3. Tell the user what to pick in the installer:
   - **Components:** PostgreSQL Server and Command Line Tools are required. pgAdmin 4 is optional (a handy GUI). Leave **Stack Builder unchecked**.
   - **Data directory:** keep the default, `C:\Program Files\PostgreSQL\18\data`.
   - **Superuser password:** the user chooses it, stores it in a password manager, and won't share it with you.
   - **Port:** `5432`. **Locale:** default.
4. Add the tools to the user PATH, then open a new shell:
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path","User") + ";C:\Program Files\PostgreSQL\18\bin", "User")
   ```
5. Have the user create `%APPDATA%\postgresql\pgpass.conf`. Create the folder if it's missing. It needs this line, with their real password:
   ```
   localhost:5432:*:postgres:<superuser password>
   ```
6. Make sure the service starts automatically:
   ```powershell
   Get-Service postgresql* | Select-Object Name, Status, StartType
   ```
   The expected name is `postgresql-x64-18`. That name comes from a third-party source, so use whatever name actually appears and substitute it everywhere below. If `StartType` isn't `Automatic`, run `Set-Service <name> -StartupType Automatic` (elevated).

**Completion check:** `psql -h localhost -U postgres -c "select version();"` prints PostgreSQL 18.x without asking for a password, and the service is `Running` and `Automatic`.

---

### Phase 2: Inspect the Supabase database (read only)

1. The Supabase project may be **paused**. If connecting times out or returns a "tenant/project not found" error, ask the user to open the Supabase dashboard and restore the project, then retry.
2. Have the user add this line to `pgpass.conf`. If they don't know the database password, they can reset it in Supabase under Project Settings → Database. That doesn't affect API-key clients.
   ```
   aws-0-us-west-2.pooler.supabase.com:5432:postgres:postgres.augusbsnxlezrehaksfx:<supabase db password>
   ```
3. Define the connection string once. It contains no secret:
   ```powershell
   $SUPA = "host=aws-0-us-west-2.pooler.supabase.com port=5432 dbname=postgres user=postgres.augusbsnxlezrehaksfx sslmode=require"
   ```
4. Work in a scratch folder outside the repo, for example `C:\Users\arete\pg_migration\`. Then gather:
   ```powershell
   psql $SUPA -c "select version();"
   psql $SUPA -c "\dt public.*"
   psql $SUPA -c "select pg_size_pretty(pg_database_size(current_database()));"
   psql $SUPA -c "select relname, n_live_tup, pg_size_pretty(pg_total_relation_size(relid)) from pg_stat_user_tables where schemaname='public' order by pg_total_relation_size(relid) desc;"
   ```
5. **Get exact row counts.** This is the baseline for Phase 5. Save the output to `counts_supabase.txt`:
   ```sql
   select 'companies' t, count(*) from public.companies union all
   select 'documents', count(*) from public.documents union all
   select 'keyword_sets', count(*) from public.keyword_sets union all
   select 'exposure_runs', count(*) from public.exposure_runs union all
   select 'keyword_matches', count(*) from public.keyword_matches;
   ```
6. **Dump the live schema** and compare it with Appendix A:
   ```powershell
   pg_dump $SUPA --schema-only --schema=public --no-owner --no-privileges -f supabase_public_schema.sql
   ```
   Look for tables, columns, constraints or column defaults that aren't in Appendix A. If you find any, add them to `db/schema.sql` in Phase 4 (minus the Supabase-only pieces from section 2), and tell the user what was added.
7. **Check for duplicate NULL-ticker companies**:
   ```sql
   select name, count(*) from public.companies where ticker is null group by name having count(*) > 1;
   ```
   Note the result. It decides which path Phase 5 takes.

**Completion check:** you have `counts_supabase.txt` and `supabase_public_schema.sql`, every difference from Appendix A is listed (or confirmed to be none), the duplicate-check result is recorded, and you've reported all of this to the user.

---

### Phase 3: Configure Postgres and the network

1. **Server settings.** Use `ALTER SYSTEM`, which writes `postgresql.auto.conf`, instead of hand-editing `postgresql.conf`:
   ```sql
   ALTER SYSTEM SET listen_addresses = '*';
   ALTER SYSTEM SET max_connections = 30;
   ALTER SYSTEM SET shared_buffers = '1GB';
   ALTER SYSTEM SET effective_cache_size = '8GB';
   ALTER SYSTEM SET work_mem = '16MB';
   ALTER SYSTEM SET maintenance_work_mem = '256MB';
   ALTER SYSTEM SET password_encryption = 'scram-sha-256';
   ```
   Run them as `psql -h localhost -U postgres -c "..."`. What each one does:
   - `shared_buffers = 1GB` holds the whole database in memory.
   - `effective_cache_size` is only a hint to the query planner; it allocates nothing.
   - `max_connections = 30` caps worst-case memory and still leaves room for a connection pool, pgAdmin and a few collaborators.
2. **pg_hba.conf** is at `C:\Program Files\PostgreSQL\18\data\pg_hba.conf`. Editing it needs admin rights. Back it up first as `pg_hba.conf.bak`, then append:
   ```
   # Tailscale only (IPv4 CGNAT range + Tailscale IPv6 ULA range); app roles only, never postgres
   host    research    research_app,research_ro    100.64.0.0/10           scram-sha-256
   host    research    research_app,research_ro    fd7a:115c:a1e0::/48     scram-sha-256
   ```
   - Keep the existing `127.0.0.1/32` and `::1/128` lines.
   - If any line allows `0.0.0.0/0`, `::/0` or `all` from a non-loopback range, show it to the user and ask before removing it.
   - The IPv6 line matters because MagicDNS names like `desktop-64ns9eu` resolve to both the IPv4 and IPv6 tailnet addresses, and clients may prefer IPv6.
3. **Restart** to apply `listen_addresses` and `shared_buffers` (elevated): `Restart-Service postgresql-x64-18`. Then confirm both settings and the rules:
   ```sql
   show listen_addresses; show shared_buffers;
   select line_number, type, database, user_name, address, error from pg_hba_file_rules;
   ```
   No row should have an `error`.
4. **Firewall. GATE.**
   1. Look first:
      ```powershell
      Get-NetAdapter | Select-Object Name, InterfaceDescription, Status
      Get-NetFirewallRule | Where-Object DisplayName -like '*postgres*' | Select-Object DisplayName, Enabled, Direction, Action
      ```
      Find the Tailscale adapter's actual `Name`, usually `Tailscale`. If the installer created a broad rule for 5432, show it to the user and propose disabling it with `Disable-NetFirewallRule`, which is reversible.
   2. Then add the rule (elevated):
      ```powershell
      New-NetFirewallRule -DisplayName "PostgreSQL via Tailscale" -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort 5432 -InterfaceAlias "Tailscale" `
        -RemoteAddress 100.64.0.0/10, fd7a:115c:a1e0::/48 -Profile Any
      ```
   3. Also confirm that Tailscale's "Allow incoming connections" is on (shields-up is off). Run `tailscale status`; it shouldn't mention shields up.

**Completion check:** after the restart, `show listen_addresses` returns `*` and `show shared_buffers` returns `1GB`. `pg_hba_file_rules` has no errors and includes both Tailscale lines. Exactly one enabled inbound rule allows 5432, limited to the Tailscale ranges and adapter.

---

### Phase 4: Create the database, roles and schema

1. **Create the database and roles** as `postgres`:
   ```sql
   CREATE DATABASE research;
   CREATE ROLE research_app LOGIN;
   CREATE ROLE research_ro  LOGIN;
   REVOKE CONNECT ON DATABASE research FROM PUBLIC;
   GRANT CONNECT ON DATABASE research TO research_app, research_ro;
   ```
2. **Set the passwords.** Have the user open `psql -h localhost -U postgres -d research` themselves and run `\password research_app` and `\password research_ro`. They choose and store the passwords. Wait until they say it's done.
3. **Write the schema file** to `db/schema.sql` in the repo. Start from Appendix A, plus any additions found in Phase 2.
   - If Phase 2 found duplicate NULL-ticker companies, change the companies constraint in the file to plain `UNIQUE (name, ticker)` for now. Phase 5 switches it after deduplicating.
   - Apply it:
     ```powershell
     psql -h localhost -U postgres -d research -v ON_ERROR_STOP=1 -f db/schema.sql
     ```
4. **Grant permissions.** Save this as `db/roles.sql` and run it the same way:
   ```sql
   GRANT USAGE ON SCHEMA public TO research_app, research_ro;
   GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO research_app;
   GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO research_app;
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO research_ro;
   ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO research_app;
   ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO research_app;
   ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT ON TABLES TO research_ro;
   ```
   `postgres` owns the tables and runs all future schema changes. The app roles can't change the schema.

**Completion check:** `psql -h localhost -U postgres -d research -c "\dt"` lists the 5 tables (plus any Phase 2 additions). `\dp` shows `research_app` with arwd and `research_ro` with r on every table. Both roles have passwords set (the user confirms).

---

### Phase 5: Copy the data from Supabase

1. **Dump the data only**, in custom format, public schema only:
   ```powershell
   pg_dump $SUPA --data-only --schema=public --no-owner --no-privileges -Fc -f supabase_public_data.dump
   ```
   - The data comes out as `COPY` statements. COPY writes the given values into `GENERATED ALWAYS` identity columns, so ids are preserved.
   - The dump also includes `setval` for each identity sequence. Those apply because the local sequences have the same default names (`<table>_id_seq`).
2. **Restore** as the superuser:
   ```powershell
   pg_restore -h localhost -U postgres -d research --data-only --no-owner --disable-triggers --single-transaction --exit-on-error supabase_public_data.dump
   ```
   - `--disable-triggers` stops foreign-key checks from failing on the order tables load in. It needs superuser, which `postgres` is.
   - `--single-transaction` means a failure leaves the tables empty, not half-filled. Fix the cause, then rerun.
3. **Verify:**
   - Run the row-count query from Phase 2 locally and compare it with `counts_supabase.txt`. Every table must match exactly.
   - Check that each sequence is ahead of its data. For each table, `select max(id) from <t>;` must be ≤ `select last_value from <t>_id_seq;`. If one isn't, run `select setval(pg_get_serial_sequence('<t>','id'), (select max(id) from <t>));`.
   - Run `ANALYZE;`.
4. **Only if Phase 2 found duplicate NULL-ticker companies. GATE: this changes data.**
   1. Show the user the duplicate list and the plan: point every document at the lowest company id per name, then delete the other copies.
   2. After they agree:
      ```sql
      BEGIN;
      CREATE TEMP TABLE dup AS
        SELECT id, min(id) OVER (PARTITION BY name) AS keep_id
        FROM companies WHERE ticker IS NULL;
      UPDATE documents d SET company_id = dup.keep_id FROM dup WHERE d.company_id = dup.id AND dup.id <> dup.keep_id;
      DELETE FROM companies c USING dup WHERE c.id = dup.id AND dup.id <> dup.keep_id;
      ALTER TABLE companies DROP CONSTRAINT companies_name_ticker_key,
        ADD CONSTRAINT companies_name_ticker_key UNIQUE NULLS NOT DISTINCT (name, ticker);
      COMMIT;
      ```
   3. Update `db/schema.sql` to the `NULLS NOT DISTINCT` form so it matches the database. Report how many company rows were merged; the `companies` count will now be lower than Supabase's, as expected.

**Completion check:** every table's row count matches Supabase (except companies after an approved dedupe, which matches the expected lower number). No sequence is behind its data. `\d companies` shows `UNIQUE NULLS NOT DISTINCT (name, ticker)`.

---

### Phase 6: Back up and keep the machine available

1. **Backup script.** Write `db/backup.ps1`:
   - Run `pg_dump -h localhost -U postgres -d research -Fc -w` to `<backup dir>\research_<yyyy-MM-dd_HHmm>.dump`. `-w` means it never prompts; the password comes from pgpass.conf.
   - Delete dumps older than 14 days in that directory, and only `research_*.dump` files.
   - Exit with an error if pg_dump fails, and write a one-line log entry to `<backup dir>\backup.log`.
2. **Backup location. Ask the user.** Ideally a second drive, or a synced cloud folder such as OneDrive or Google Drive, so a dead SSD doesn't take out the backups too. The default if they have no preference is `C:\pg_backups`.
3. **Schedule it daily at 03:00. GATE.**
   ```powershell
   $action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"<repo>\db\backup.ps1`""
   $trigger = New-ScheduledTaskTrigger -Daily -At 3am
   Register-ScheduledTask -TaskName "Postgres research backup" -Action $action -Trigger $trigger
   ```
   This registers it to run only while the user is logged on. If they want it to run while logged off too, they set "Run whether user is logged on or not" in Task Scheduler, which asks for their Windows password; they enter it themselves. The task must run as `arete` so it finds that user's `pgpass.conf`.
4. **Test it:** run `db/backup.ps1` once by hand. Then prove the dump restores into a throwaway database:
   ```powershell
   psql -h localhost -U postgres -c "CREATE DATABASE research_restore_test;"
   pg_restore -h localhost -U postgres -d research_restore_test --no-owner <dump>
   ```
   Compare row counts with `research`. When you ask to schedule backups, also ask to drop `research_restore_test` after the test, then drop it with `DROP DATABASE research_restore_test;`.
5. **Availability. GATE: these are system settings, so present them together and let the user choose.**
   - Don't sleep or hibernate on AC power: `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0`. The display may still turn off; that's fine.
   - Tailscale: Run unattended on (tray → Preferences, or `tailscale set --unattended=true`). Otherwise Tailscale drops when the user logs out.
   - Things the user does themselves, outside the machine:
     - Tailscale admin console → Machines → `desktop-64ns9eu` → **Disable key expiry**.
     - Windows Settings → Windows Update → Active hours.
     - Optionally, in BIOS, "Restore on AC power loss" set to On.
   - Windows Update will still occasionally restart the machine. That's fine: the service is Automatic and Tailscale is unattended, so everything comes back without anyone logging in. Postgres's write-ahead log makes a sudden power-off safe, provided `fsync` stays on (the default; never turn it off).

**Completion check:** a backup file exists, it restored into the test database with matching counts, the scheduled task is registered (`Get-ScheduledTask "Postgres research backup"`), and the user has chosen on each availability item.

---

### Phase 7: Code — add `PostgresUploader`

Work on a new branch, for example `feat/postgres-uploader`. Read `src/analysis/supabase_uploader.py` and `process_batch_to_supabase()` in `src/analysis/batch_runner.py` in full first, and match their style: docstrings, type hints, comment density.

1. **New file `src/analysis/postgres_uploader.py`**, class `PostgresUploader`. It's a copy of `SupabaseUploader` with the same public interface and the same return dicts:
   - `__init__(self, database_url: str, batch_size: int = 500, max_connections: int = 4)` creates a `psycopg_pool.ConnectionPool`.
   - `initialize_company_cache()` loads `id, name, ticker` for every company with one query; no paging needed.
   - `upload_document(data, year)`: same field mapping as `SupabaseUploader.upload_document`, and the same statuses: `success` / `skipped` / `error` with `reason`.
   - `finalize()` flushes any buffer and closes the pool.
   - Copy the helpers `_parse_date`, `_safe_int` and `_safe_float` as they are.
2. **Leave `supabase_uploader.py` unchanged.**
3. **SQL behaviour, matching what Supabase did:**
   - Company get-or-create:
     ```sql
     INSERT INTO companies (name, ticker) VALUES (%s, %s)
     ON CONFLICT (name, ticker) DO UPDATE SET name = EXCLUDED.name
     RETURNING id
     ```
     The no-op `DO UPDATE` makes `RETURNING` give back the id of an existing row too. This works together with the `NULLS NOT DISTINCT` constraint. Simpler than the Supabase version: insert the company immediately when it misses the cache, with no buffering. Guard the cache with a `threading.Lock`.
   - Document upsert: `INSERT INTO documents (<the doc_row columns>) VALUES (...) ON CONFLICT (source_file) DO UPDATE SET <each doc_row column except source_file> = EXCLUDED.<col> RETURNING id`. Update **only** the columns in `doc_row`, so columns written elsewhere (`rc_*`, `re_*`, `rx_*`, `risk_total_sentences`) are kept. That's what the Supabase upsert does today.
   - Pass values as query parameters only, never with string formatting. Column names come from the fixed `doc_row` keys, so building the column list from them is safe.
4. **Concurrency:** the batch runner calls `upload_document` from several threads, and a psycopg connection can't be shared across threads. Each call should therefore `with self.pool.connection() as conn:`, which commits on a clean exit and rolls back on an exception.
5. **Resilience:** on `psycopg.OperationalError` (server restarting, Tailscale blip), retry up to 3 times with backoff of 1 s, 3 s, then 10 s, then return `{"status": "error", ...}`. Retrying is safe because both statements are idempotent upserts.
6. **`BatchRunner`:**
   - Add `process_batch_to_postgres(self, input_dir, year, database_url: str | None = None, ...)` with the same parameters as `process_batch_to_supabase`, except `database_url` replaces `supabase_client` and defaults to `os.environ["DATABASE_URL"]`.
   - Size the pool to the effective thread count.
   - Don't copy the roughly 150-line body: move the shared loop into a private `_process_batch_with_uploader(uploader, ...)` used by both methods, keeping `process_batch_to_supabase`'s signature and behaviour identical.
7. **Dependencies:**
   - Add `psycopg[binary,pool]` (psycopg 3) to the pip section of `environment.yml`.
   - Add `psycopg==<installed>`, `psycopg-binary==<installed>` and `psycopg-pool==<installed>` to `requirements.txt`, keeping its UTF-16 LE encoding (section 1).
   - Install them into the env.
8. **Config:** the code reads `DATABASE_URL`. The user puts it in `.env` (gitignored) themselves:
   - On the desktop: `postgresql://research_app:<pw>@localhost:5432/research`
   - On other machines: `postgresql://research_app:<pw>@desktop-64ns9eu:5432/research`
   - Add a short "Using the local Postgres" section to `db/README.md` (create it) that shows this, with placeholders only.
9. **Test:** add `test/test_postgres_uploader.py` (pytest style). Skip it unless a `TEST_DATABASE_URL` env var is set. It should:
   - Run against a throwaway database (`research_test`, created from `db/schema.sql`), never `research`.
   - Upload the same fake document twice: expect 1 document row, with the second upload updating its fields.
   - Upload two documents for a company with `ticker=None`: expect 1 company row.
   - Upload from 4 threads at once: no errors, and the expected row counts.

   Run it. Then do one real run of `process_batch_to_postgres` on a few files from `data/earnings_calls/2016/` against `research`, and confirm the rows appear.
10. **Commit** (`db/schema.sql`, `db/roles.sql`, `db/backup.ps1`, `db/README.md`, this handoff doc, and the code). **GATE:** ask before pushing or opening a PR.

**Completion check:** the tests pass, a real batch on a few files lands in `research` with the expected rows, `supabase_uploader.py` is unchanged (`git diff` shows nothing for it), and `process_batch_to_supabase` keeps its signature.

---

### Phase 8: Hand back to the user

The last test needs the Mac, which you can't control. Give the user these to run on the Mac (psql is already installed there via Homebrew), with `research_app`'s password in the Mac's `~/.pgpass`:

```bash
psql "host=desktop-64ns9eu port=5432 dbname=research user=research_app" -c "select count(*) from documents;"
```
```bash
psql "host=desktop-64ns9eu port=5432 dbname=research user=postgres" -c "select 1;"
```

The first should return the documents count. The second **must fail** with a pg_hba "no entry" error, which proves the superuser can't connect remotely.

Finish with a summary for the user:
- What was installed, and where.
- Final row counts.
- Backup location and schedule.
- Which availability settings were applied, and which are left for them (key expiry, active hours, BIOS).
- The branch name and commit.
- Anything that didn't match this doc, and what you did about it.

---

## Appendix A: Clean schema (`db/schema.sql`)

This is the Supabase schema as of migration `20260904170135` with the Supabase-only pieces removed. Merge in anything Phase 2 found. Keep the constraint and index names; they match Supabase, so a future diff stays readable.

```sql
-- Research database schema (self-hosted Postgres 18).
-- Derived from Supabase migration 20260904170135_remote_schema.sql, minus Supabase-only objects.

CREATE TABLE public.companies (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name       text NOT NULL,
    ticker     text,
    created_at timestamptz DEFAULT now(),
    CONSTRAINT companies_name_ticker_key UNIQUE NULLS NOT DISTINCT (name, ticker)
);

CREATE TABLE public.documents (
    id                    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    company_id            bigint REFERENCES public.companies(id) ON DELETE CASCADE,
    year                  integer NOT NULL,
    source_file           text,
    event_title           text,
    event_date            timestamptz,
    city                  text,
    num_sentences         integer,
    num_words             integer,
    sentiment             double precision,
    ml_score              double precision,
    lm_score              double precision,
    hiv4_score            double precision,
    flesch_reading_ease   double precision,
    flesch_kincaid_grade  double precision,
    smog_index            double precision,
    coleman_liau_index    double precision,
    automated_readability double precision,
    dale_chall_score      double precision,
    difficult_words       integer,
    linsear_write_formula double precision,
    gunning_fog           double precision,
    text_standard         text,
    created_at            timestamptz DEFAULT now(),
    rc_active_ratio       double precision,
    rc_passive_ratio      double precision,
    re_active_ratio       double precision,
    re_passive_ratio      double precision,
    rx_active_ratio       double precision,
    rx_passive_ratio      double precision,
    risk_total_sentences  integer,
    CONSTRAINT documents_source_file_unique UNIQUE (source_file)
);

CREATE TABLE public.keyword_sets (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name           text NOT NULL,
    source_file    text,
    total_keywords integer,
    created_at     timestamptz DEFAULT now(),
    CONSTRAINT keyword_sets_name_key UNIQUE (name)
);

CREATE TABLE public.exposure_runs (
    id                          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id                 bigint REFERENCES public.documents(id) ON DELETE CASCADE,
    keyword_set_id              bigint REFERENCES public.keyword_sets(id) ON DELETE CASCADE,
    matching_method             text DEFAULT 'direct',
    cosine_threshold            double precision,
    runtime_seconds             double precision,
    total_keywords_searched     integer,
    total_keywords_with_matches integer,
    total_direct_matches        integer,
    total_cosine_matches        integer,
    total_sentences_in_document integer,
    total_sentences_with_match  integer,
    pct_sentences_with_match    double precision,
    run_timestamp               text,
    created_at                  timestamptz DEFAULT now(),
    -- Supabase had this constraint twice under two names; one is enough.
    CONSTRAINT exposure_runs_doc_kwset_unique UNIQUE (document_id, keyword_set_id)
);

CREATE TABLE public.keyword_matches (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    exposure_run_id bigint REFERENCES public.exposure_runs(id) ON DELETE CASCADE,
    keyword         text NOT NULL,
    match_type      text NOT NULL,
    matched_text    text,
    context         text,
    position        integer,
    created_at      timestamptz DEFAULT now(),
    CONSTRAINT keyword_matches_match_type_check CHECK (match_type IN ('direct', 'cosine')),
    CONSTRAINT keyword_matches_unique UNIQUE (exposure_run_id, keyword, match_type, matched_text, position)
);

CREATE INDEX idx_documents_company    ON public.documents (company_id);
CREATE INDEX idx_documents_event_date ON public.documents (event_date);
CREATE INDEX idx_documents_year       ON public.documents (year);
CREATE INDEX idx_exposure_doc         ON public.exposure_runs (document_id);
CREATE INDEX idx_exposure_kw          ON public.exposure_runs (keyword_set_id);
CREATE INDEX idx_matches_keyword      ON public.keyword_matches (keyword);
CREATE INDEX idx_matches_run          ON public.keyword_matches (exposure_run_id);
```

Two facts about the data:
- No code in this repo writes `keyword_sets`, `exposure_runs` or `keyword_matches`. They were filled by some other process, so for them the job is just migrating the data.
- `keyword_matches_unique` has the same NULLs-are-distinct issue as companies (nullable `matched_text` and `position`). Leave it as is; mention it to the user as a possible follow-up.

---

## Appendix B: Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `psql` not found | PATH not updated, or the shell wasn't reopened (Phase 1, step 4). Or call `& "C:\Program Files\PostgreSQL\18\bin\psql.exe"` directly. |
| `fe_sendauth: no password supplied` | pgpass.conf is missing, in the wrong folder, or a field doesn't match. It must be `%APPDATA%\postgresql\pgpass.conf` for the user running the command, with host, port, db and user matching exactly. `*` works as a wildcard. |
| Supabase connection times out | Project paused (restore it in the dashboard), or you're using the direct `db.*` host, which is IPv6-only. Use the pooler host from section 1. |
| `pg_dump: error: aborting because of server version mismatch` | The local pg_dump is older than the server. Use the PG 18 binaries. |
| `no pg_hba.conf entry for host "100.x..."` | The pg_hba line is missing, or the database/user doesn't match it. Check `pg_hba_file_rules`. pg_hba changes take effect for new connections without a restart. |
| Remote connection times out (no error) | Firewall: wrong `-InterfaceAlias`, rule disabled, or Tailscale shields-up is on. Or `listen_addresses` didn't take effect (it needs a restart). |
| Restore fails on duplicate key | Tables weren't empty. With `--single-transaction` nothing was committed: `TRUNCATE companies, documents, keyword_sets, exposure_runs, keyword_matches RESTART IDENTITY CASCADE;` (this is a delete, so it's a GATE), then restore again. |
| Restore fails on a role or extension | The dump wasn't made with `--data-only --no-owner --no-privileges`. Redo Phase 5, step 1. |
| Everything is slow while the user games | Expected only during big uploads. Postgres can't be throttled per process easily; schedule heavy batch runs for when the user isn't playing. |

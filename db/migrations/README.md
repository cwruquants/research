# Schema migrations

One SQL file per schema change made after the initial migration, applied in
filename order: `YYYYMMDD_HHMM_<what>.sql`, for example
`20261020_1430_add_documents_notes.sql`.

Each file starts with a comment saying what the change is for and who or what
made it, and is safe to read on its own. Fold the same change into
`../tables.sql` or `../constraints.sql` in the same commit, so `../schema.sql`
always builds the current schema from scratch.

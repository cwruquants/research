-- Complete local schema. For a bulk migration, apply tables.sql before the
-- data restore and constraints.sql afterward to avoid maintaining indexes
-- during the 37-million-row COPY.
\ir tables.sql
\ir constraints.sql

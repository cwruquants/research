BEGIN;
-- Superuser-owned schema for the guard functions; the agent can't change anything in it.
CREATE SCHEMA admin AUTHORIZATION postgres;
REVOKE ALL ON SCHEMA admin FROM PUBLIC;

-- Blocks TRUNCATE by research_agent unless the transaction set research.allow_drop = on.
CREATE FUNCTION admin.guard_truncate() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF session_user = 'research_agent'
     AND coalesce(current_setting('research.allow_drop', true), '') <> 'on' THEN
    RAISE EXCEPTION 'research_agent may not TRUNCATE %.% without the user''s approval', TG_TABLE_SCHEMA, TG_TABLE_NAME
      USING HINT = 'After the user approves, run it in a transaction that first does SET LOCAL research.allow_drop = on.';
  END IF;
  RETURN NULL;
END $$;
-- The agent creates tables, and the event trigger below attaches this guard to them as the agent.
GRANT USAGE ON SCHEMA admin TO research_agent;
GRANT EXECUTE ON FUNCTION admin.guard_truncate() TO research_agent;

-- Blocks every drop by research_agent (tables, columns, constraints, indexes, triggers, ...) unless approved.
CREATE FUNCTION admin.guard_agent_drops() RETURNS event_trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
  IF session_user = 'research_agent'
     AND coalesce(current_setting('research.allow_drop', true), '') <> 'on'
     AND EXISTS (SELECT 1 FROM pg_event_trigger_dropped_objects() WHERE NOT is_temporary) THEN
    RAISE EXCEPTION 'research_agent may not drop objects without the user''s approval (%)', tg_tag
      USING HINT = 'After the user approves, run it in a transaction that first does SET LOCAL research.allow_drop = on.';
  END IF;
END $$;
CREATE EVENT TRIGGER guard_agent_drops ON sql_drop EXECUTE FUNCTION admin.guard_agent_drops();

-- After the agent's DDL: give new tables the TRUNCATE guard, and refuse any attempt to disable the guard.
CREATE FUNCTION admin.guard_agent_ddl() RETURNS event_trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
DECLARE r record;
BEGIN
  IF session_user <> 'research_agent' THEN RETURN; END IF;
  FOR r IN SELECT c.objid FROM pg_event_trigger_ddl_commands() c
           JOIN pg_class t ON t.oid = c.objid
           WHERE c.command_tag IN ('CREATE TABLE', 'CREATE TABLE AS', 'SELECT INTO')
             AND c.object_type = 'table' AND t.relpersistence <> 't'
  LOOP
    EXECUTE format('CREATE TRIGGER zz_guard_truncate BEFORE TRUNCATE ON %s FOR EACH STATEMENT EXECUTE FUNCTION admin.guard_truncate()', r.objid::regclass);
  END LOOP;
  IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgfoid = 'admin.guard_truncate'::regproc AND tgenabled <> 'O') THEN
    RAISE EXCEPTION 'research_agent may not disable the TRUNCATE guard';
  END IF;
END $$;
CREATE EVENT TRIGGER guard_agent_ddl ON ddl_command_end EXECUTE FUNCTION admin.guard_agent_ddl();

-- Guard the existing tables.
CREATE TRIGGER zz_guard_truncate BEFORE TRUNCATE ON public.companies       FOR EACH STATEMENT EXECUTE FUNCTION admin.guard_truncate();
CREATE TRIGGER zz_guard_truncate BEFORE TRUNCATE ON public.documents       FOR EACH STATEMENT EXECUTE FUNCTION admin.guard_truncate();
CREATE TRIGGER zz_guard_truncate BEFORE TRUNCATE ON public.exposure_runs   FOR EACH STATEMENT EXECUTE FUNCTION admin.guard_truncate();
CREATE TRIGGER zz_guard_truncate BEFORE TRUNCATE ON public.keyword_matches FOR EACH STATEMENT EXECUTE FUNCTION admin.guard_truncate();
CREATE TRIGGER zz_guard_truncate BEFORE TRUNCATE ON public.keyword_sets    FOR EACH STATEMENT EXECUTE FUNCTION admin.guard_truncate();
COMMIT;

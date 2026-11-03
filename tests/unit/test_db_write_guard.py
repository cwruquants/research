"""The Claude Code hook that asks before destructive SQL (.claude/hooks/db_write_guard.py)."""

import importlib.util
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "db_write_guard.py"
spec = importlib.util.spec_from_file_location("db_write_guard", HOOK)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

MODIFY = "mcp__research-db__postgres_mcp_modify"


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO public.keyword_sets (name) VALUES ('risk')",
        "UPDATE public.documents SET year = 2017 WHERE id = 1",
        "UPDATE public.documents SET notes = 'a;b' WHERE id = 1",
        "INSERT INTO t (id) VALUES (1) ON CONFLICT (id) DO UPDATE SET id = excluded.id",
        "ALTER TABLE public.documents ADD COLUMN notes text",
        "UPDATE public.documents SET notes = E'it\\'s' WHERE id = 1 AND kind = 'a'",
        'UPDATE public."o\'neil" SET a = 1 WHERE id = 1',
        "UPDATE public.documents SET year = 1 WHERE id IN (SELECT id FROM public.t)",
        "UPDATE public.documents SET notes = $tag$ ; $tag$ WHERE id = 1",
        "WITH fix AS (SELECT id FROM public.t), done AS (UPDATE public.documents d SET year = 1 "
        "FROM fix WHERE d.id = fix.id RETURNING d.id) SELECT count(*) FROM done",
        "CREATE TABLE public.scratch (id int)",
    ],
)
def test_routine_writes_run(sql):
    assert guard.decide(MODIFY, {"statement": sql}) == ("allow", [])


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE public.scratch",
        "DELETE FROM public.documents WHERE id = 1",
        "UPDATE public.documents SET year = 2017",
        "UPDATE public.documents SET notes = 'where'",
        "UPDATE public.documents SET year = 2017 -- where clause left out",
        "UPDATE public.documents SET year = 2017 /* where */",
        "UPDATE public.documents SET notes = $$ where $$",
        "UPDATE public.documents SET year = 1 WHERE id = 1; UPDATE public.companies SET name = 'x'",
        'UPDATE "documents" SET year = 2017',
        "UPDATE public.documents SET notes = E'it\\'s where it was'",
        "UPDATE public.documents SET year = 2017 /* old /* note */ where */",
        "UPDATE public.\"o'neil\" SET a = 1 WHERE id = 1; UPDATE public.companies SET name = 'x'",
        "UPDATE public.documents d SET year = (SELECT max(y) FROM public.t WHERE t.id = d.id)",
        "WITH s AS (SELECT id FROM public.t WHERE x) UPDATE public.documents SET year = 1",
        "WITH done AS (UPDATE public.documents SET year = 1 RETURNING id) SELECT * FROM done WHERE id > 0",
        "UPDATE public.documents SET notes = 'unterminated where",
        "ALTER TABLE public.documents ALTER COLUMN num_words TYPE smallint",
        "ALTER TABLE public.documents ALTER num_words TYPE smallint",
        'ALTER TABLE public.documents ALTER "num_words" SET DATA TYPE smallint',
        "GRANT ALL ON public.documents TO PUBLIC",
        "",
    ],
)
def test_destructive_or_unreadable_sql_asks(sql):
    decision, reasons = guard.decide(MODIFY, {"statement": sql})
    assert decision == "ask" and reasons


def test_bulk_load_always_asks():
    assert guard.decide("mcp__research-db__postgres_mcp_bulk_load_csv", {})[0] == "ask"

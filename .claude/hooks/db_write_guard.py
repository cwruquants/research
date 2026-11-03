"""PreToolUse hook for the research-db MCP write tools.

Routine writes and schema changes run without a prompt. Anything that removes
data or objects, or could hide or enable such a change, is sent to the user for
approval with the reason shown. Fails closed: input it can't read is also sent
for approval.

The database refuses drops and truncates by research_agent on its own (see
db/agent_guards.sql). This hook is what turns that refusal into an approval
prompt, and it also covers DELETE, permission changes and lossy type changes,
which the database does not block.

Matching runs on the raw SQL, comments and string literals included, so a
destructive word in data or a comment causes an extra prompt, never a missed one.
"""

import json
import os
import re
import sys
import time

# Every decision is appended here, so it's possible to check afterwards what ran and why.
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_write_guard.log")

ASK_PATTERNS = [
    (r"\bdrop\b", "DROP removes an object or column"),
    (r"\btruncate\b", "TRUNCATE empties a table"),
    (r"\bdelete\b", "DELETE removes rows"),
    (r"allow_drop", "sets the drop-approval flag"),
    (r"\bset_config\b", "set_config can set the drop-approval flag"),
    (r"(?:^|;)\s*do\b", "DO block runs arbitrary code"),
    (r"\bexecute\b", "EXECUTE runs dynamically built SQL"),
    (r"\bcall\b", "CALL runs a procedure"),
    (r"\bcopy\b", "COPY reads or writes files or bulk data"),
    (r"\bcreate\s+(?:or\s+replace\s+)?(?:constraint\s+)?(?:function|procedure|trigger|rule|event\s+trigger|aggregate|operator|extension|server|foreign)\b",
     "creates code or server objects that could run later"),
    (r"\bdisable\b", "DISABLE turns off triggers or row security"),
    (r"\b(?:grant|revoke)\b", "changes permissions"),
    (r"\bowner\s+to\b", "changes ownership"),
    (r"\bsession\s+authorization\b|\b(?:set|reset)\s+role\b", "switches roles"),
    (r"\balter\s+(?:role|user|group|system|database|default\s+privileges|event\s+trigger|extension|server|foreign)\b",
     "changes server-level settings"),
    (r"\balter\s+column\b[^;]*\btype\b", "column type change can lose data"),
    (r"\bquery_to_xml\b|\bdblink\w*\b", "runs SQL passed as a string"),
]


def unbounded_updates(sql):
    """Statements that UPDATE a table without a WHERE clause rewrite every row.

    A table name must follow UPDATE, which leaves out upserts (ON CONFLICT DO UPDATE SET).
    """
    update = re.compile(r"\bupdate\s+(?:only\s+)?(?!set\b)[\w.\"]+[\s\S]*?\bset\b", re.I)
    return [s for s in sql.split(";") if update.search(s) and not re.search(r"\bwhere\b", s, re.I)]


def decide(tool_name, tool_input):
    if tool_name.endswith("postgres_mcp_bulk_load_csv"):
        return "ask", ["bulk CSV load can truncate or overwrite a table"]
    sql = tool_input.get("statement")
    if not isinstance(sql, str) or not sql.strip():
        return "ask", ["could not read the SQL statement"]
    reasons = [why for pattern, why in ASK_PATTERNS if re.search(pattern, sql, re.I)]
    if unbounded_updates(sql):
        reasons.append("UPDATE without WHERE rewrites every row")
    return ("ask", reasons) if reasons else ("allow", [])


def main():
    event = {}
    try:
        event = json.load(sys.stdin)
        decision, reasons = decide(event.get("tool_name", ""), event.get("tool_input") or {})
    except Exception as exc:  # fail closed
        decision, reasons = "ask", [f"hook could not parse its input ({exc.__class__.__name__})"]
    reason = ("Needs your approval: " + "; ".join(reasons)) if decision == "ask" else "Routine database write"
    try:
        with open(LOG, "a") as log:
            log.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "session": event.get("session_id"),
                "permission_mode": event.get("permission_mode"),
                "tool": event.get("tool_name"),
                "decision": decision,
                "reasons": reasons,
                "statement": (event.get("tool_input") or {}).get("statement"),
            }) + "\n")
    except OSError:
        pass  # logging must never change the decision
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason,
    }}))


if __name__ == "__main__":
    main()

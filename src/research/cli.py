"""`research`: one command for every pipeline step. Run `research <step> --help` for options."""

import argparse
import importlib
import runpy
import sys

from research import config

# command path -> (module whose main() runs it, one-line summary)
COMMANDS = {
    ("transcripts", "load"): (
        "research.pipelines.load_transcript_files",
        "Store the raw transcript XMLs from the vendor zips in transcript_files",
    ),
    ("transcripts", "verify"): (
        "research.pipelines.verify_transcript_files",
        "Check transcript_files against the zips it was loaded from",
    ),
    ("ingest",): ("research.pipelines.ingest", "Parse transcript XMLs into companies and documents rows"),
    ("backfill",): ("research.pipelines.backfill_exposures", "Write missing keyword exposure runs"),
    ("link", "route-a"): ("research.linking.route_a", "Link documents to CRSP PERMNOs by RIC and call date"),
    ("link", "route-b"): (
        "research.linking.route_b",
        "Link documents to CRSP by event title (acquirer/host rows)",
    ),
    ("link", "coverage"): ("research.linking.cusip_coverage", "Report how many DB tickers map to a CUSIP"),
    ("returns", "fetch"): ("research.returns.fetch_crsp", "Pull CRSP daily files from WRDS"),
    ("returns", "compute"): ("research.returns.event_returns", "Compute event-window returns"),
    ("returns", "mcap"): ("research.returns.market_cap", "Pre-call market cap (mcap_pre) from WRDS"),
    ("returns", "surprise"): ("research.returns.earnings_surprise", "I/B/E/S earnings surprise (SUE)"),
    ("export",): ("research.db.export", "Stream a read-only query to a CSV file"),
}


def _usage() -> str:
    lines = ["usage: research <step> [options]", "", "steps:"]
    for path, (_, summary) in COMMANDS.items():
        lines.append(f"  {' '.join(path):<18} {summary}")
    lines.append(f"  {'demo':<18} Run the onboarding demo on a synthetic transcript")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return
    if argv[0] == "demo":
        sys.argv = ["research demo", *argv[1:]]
        runpy.run_path(str(config.PROJECT_ROOT / "examples" / "onboarding_demo.py"), run_name="__main__")
        return
    for depth in (2, 1):
        path = tuple(argv[:depth])
        if path in COMMANDS:
            module = importlib.import_module(COMMANDS[path][0])
            sys.argv = [f"research {' '.join(path)}", *argv[depth:]]
            module.main()
            return
    group = [p for p in COMMANDS if p[0] == argv[0]]
    if group:
        sys.exit(f"research {argv[0]}: choose one of {', '.join(p[1] for p in group)}")
    raise SystemExit(
        argparse.ArgumentParser(prog="research").error(f"unknown step {argv[0]!r}\n\n{_usage()}")
    )


if __name__ == "__main__":
    main()

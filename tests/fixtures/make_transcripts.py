"""Write the synthetic transcript fixtures in tests/fixtures/transcripts/.

The files copy the layout of the vendor XMLs (header block, participant lists,
Presentation and Q&A sections framed by ==== and ---- rules) with invented text,
so tests and the demo never need licensed transcripts. Re-run after editing:

    uv run python tests/fixtures/make_transcripts.py
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent / "transcripts"
BAR, RULE = "=" * 80, "-" * 80

PRESENTATION = [
    (
        "Operator   [1]",
        "Good day and welcome to the Example Industries fourth quarter earnings call. "
        "Today's call is being recorded.",
    ),
    (
        "Dana Park, Example Industries Inc - CEO   [2]",
        "Thank you, and good morning. Before we start, our remarks include forward-looking statements, "
        "and actual results may differ materially from those described. "
        "Revenue grew 8% this quarter on strong demand for our pumps. "
        "Trade policy remains the main risk we see, and new taxes on imported steel could raise our costs. "
        "We have a plan to manage that risk with a second supplier.",
    ),
    (
        "Lee Moran, Example Industries Inc - CFO   [3]",
        "Gross margin improved to 34%. We expect government infrastructure spending to support orders next year, "
        "although the timing is uncertain.",
    ),
]
QA = [
    ("Operator   [4]", "Our first question comes from Sam Ortiz with Harbor Securities."),
    (
        "Sam Ortiz, Harbor Securities - Analyst   [5]",
        "Thanks. How exposed are you if the trade war gets worse?",
    ),
    (
        "Dana Park, Example Industries Inc - CEO   [6]",
        "About a fifth of our steel is imported. A broader trade war would be a real risk, "
        "but our risk management team has contracts in place through the summer.",
    ),
    ("Sam Ortiz, Harbor Securities - Analyst   [7]", "And on taxes?"),
    (
        "Lee Moran, Example Industries Inc - CFO   [8]",
        "The new taxes would cost us about two cents a share. There is little politics in our guidance; "
        "we simply assume current rules hold.",
    ),
]


def _turns(turns, tight: bool) -> str:
    gap = "" if tight else "\n"
    return "".join(f"{RULE}\n{name}\n{RULE}\n{gap}{text}\n{gap}" for name, text in turns)


def _body(tight: bool, headings: bool = True) -> str:
    header = (
        "Example Industries Inc Q4 2016 Earnings Call\n\n"
        "SPRINGFIELD Feb 2, 2017 (Example Wire) -- Edited Transcript of Example Industries Inc "
        "earnings conference call\n\nTEXT version of Transcript\n\n"
        f"{BAR}\nCorporate Participants\n{BAR}\n"
        "   *  Dana Park\n      Example Industries Inc - CEO\n"
        "   *  Lee Moran\n      Example Industries Inc - CFO\n\n"
        f"{BAR}\nConference Call Participants\n{BAR}\n"
        "   *  Sam Ortiz\n      Harbor Securities - Analyst\n\n"
    )
    if not headings:  # older layout: one "Transcript" section, no Presentation / Q&A headings
        return header + f"{BAR}\nTranscript\n{RULE}\n" + _turns(PRESENTATION + QA, tight)[len(RULE) + 1 :]
    return (
        header
        + f"{BAR}\nPresentation\n"
        + _turns(PRESENTATION, tight)[:]
        + f"{BAR}\nQuestions and Answers\n"
        + _turns(QA, tight)
        + f"{BAR}\nDefinitions\n{RULE}\n"
        + "This synthetic transcript was written for tests and contains no real call text.\n"
    )


def _xml(event_id: str, start: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        f'<Event Id="{event_id}" eventTypeId="1" eventTypeName="Earnings Conference Call">\n'
        "<eventTitle>Q4 2016 Example Industries Inc Earnings Conference Call</eventTitle>\n"
        "<city>SPRINGFIELD</city>\n<companyName>Example Industries Inc</companyName>\n"
        "<companyTicker>EXMPL.OQ</companyTicker>\n"
        f"<startDate>{start}</startDate>\n"
        f'<EventStory Id="{event_id}.F" storyType="transcript" version="Final">\n'
        "<Headline><![CDATA[Edited Transcript of EXMPL.OQ earnings conference call or presentation "
        "2-Feb-17 1:30pm GMT]]></Headline>\n"
        f"<Body>{escape(body)}</Body>\n</EventStory>\n</Event>\n"
    )


FIXTURES = {
    "spaced_T.xml": dict(event_id="9000001", start="02-Feb-17 1:30pm GMT", body=_body(tight=False)),
    "tight_T.xml": dict(event_id="9000002", start="02-Feb-17 1:30pm GMT", body=_body(tight=True)),
    "no_headings_T.xml": dict(
        event_id="9000003", start="02-Feb-17 1:30pm GMT", body=_body(False, headings=False)
    ),
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, spec in FIXTURES.items():
        (OUT / name).write_text(_xml(**spec), encoding="utf-8")
        print(f"wrote {OUT / name}")

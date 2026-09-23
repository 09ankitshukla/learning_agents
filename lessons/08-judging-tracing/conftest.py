"""pytest setup for lesson 8.

Needs more lessons on the path than earlier conftests because lesson 8 sits on top
of most of the project: lesson 1 for JSON extraction, lesson 3 for the loop and
tools, lesson 6 for the test doubles, lesson 7 for the eval runs.

That is not accidental complexity -- it is what an observability layer looks like.
Tracing and judging are cross-cutting by nature, so they touch everything.
"""

from __future__ import annotations

import sys
from pathlib import Path

LESSONS = Path(__file__).resolve().parents[1]

for _folder in (
    "01-structured-output",   # extract_json, reused by the judge's repair loop
    "02-tool-calling",        # tools, via toolset
    "03-agent-loop",          # run_agent, Trajectory
    "06-testing",             # ScriptedClient and friends
    "07-evaluation",          # EvalRun, dataset
    "08-judging-tracing",     # this lesson
):
    _path = str(LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

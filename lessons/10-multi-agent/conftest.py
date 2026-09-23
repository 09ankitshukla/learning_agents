"""pytest setup for lesson 10.

Reaches back further than any previous conftest, and the reason is worth noticing:
lesson 10 adds no new capability. It rewires ones that already exist — lesson 2's
registry and dispatcher, lesson 3's loop, lesson 4's resumable message list, lesson 7's
harness, lesson 8's spans, lesson 9's decision rule. A lesson whose conftest imports
everything is usually a lesson about composition.
"""

from __future__ import annotations

import sys
from pathlib import Path

LESSONS = Path(__file__).resolve().parents[1]

for _folder in (
    "01-structured-output",
    "02-tool-calling",
    "03-agent-loop",       # run_agent, Trajectory
    "06-testing",          # ScriptedClient, so the team can be tested offline
    "07-evaluation",       # dataset, scorers, harness
    "08-judging-tracing",  # Span, cost
    "09-iteration",        # decide, Changelog
    "10-multi-agent",      # this lesson
):
    _path = str(LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

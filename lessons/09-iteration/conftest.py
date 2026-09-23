"""pytest setup for lesson 9.

Lesson 9 is the second cross-cutting lesson after 8, and for the same reason: it
does not add a capability, it adds a way of using the ones already built. So it
imports from almost everywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

LESSONS = Path(__file__).resolve().parents[1]

for _folder in (
    "01-structured-output",
    "02-tool-calling",
    "03-agent-loop",
    "06-testing",
    "07-evaluation",       # dataset, harness, scorers -- the instrument
    "08-judging-tracing",  # cost per success
    "09-iteration",        # this lesson
):
    _path = str(LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

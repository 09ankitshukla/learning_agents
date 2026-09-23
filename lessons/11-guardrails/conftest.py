"""pytest setup for lesson 11."""

from __future__ import annotations

import sys
from pathlib import Path

LESSONS = Path(__file__).resolve().parents[1]

for _folder in (
    "02-tool-calling",
    "03-agent-loop",       # run_agent, the sandbox, the six tools
    "06-testing",          # ScriptedClient, so injection can be tested offline
    "07-evaluation",       # the dataset the injection cases were added to
    "11-guardrails",       # this lesson
):
    _path = str(LESSONS / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

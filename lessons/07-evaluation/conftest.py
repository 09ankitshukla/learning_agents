"""pytest setup for lesson 7's own tests.

Small and local on purpose. Lesson 6's conftest handles lessons 1-5; this one only
needs to make lesson 7's modules importable so its scorers can be tested.

Worth stating why lesson 7 has tests at all: the scorers are pure functions that
decide every number the harness reports. A silent bug in `normalise` or
`extract_numbers` would not crash anything -- it would quietly mark correct answers
wrong, and you would go and "fix" an agent that was working. That is the highest-
consequence, lowest-visibility kind of bug in a measurement tool, and it is exactly
what lesson 6 argued deserves tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

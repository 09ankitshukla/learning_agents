"""pytest configuration: make the earlier lessons importable, and register markers.

The awkward practical problem: lesson folders are named `03-agent-loop`, which is
not a valid Python identifier, so they cannot be packages and `import loop` does
not work from anywhere else. Every lesson from 3 onward has worked around this
with a `sys.path.insert`, and the tests need the same thing for all of them at
once.

This is a genuine cost of organising code by lesson rather than as a package, and
it is worth naming rather than hiding. In a real project you would have
`src/myagent/loop.py` and import it normally. Here the folder names carry teaching
value -- a reader can see the progression at a glance -- so the awkwardness is
paid here, in one place.

One collision to know about: lessons 2, 3, 4 and 5 all contain an `agent.py`. With
several lesson directories on `sys.path`, a bare `import agent` would resolve to
whichever came first. The tests therefore never import `agent`; they import the
library modules (`tools`, `loop`, `toolset`, `context`, `session`, `chunking`,
`store`, `queries`), whose names are unique across lessons. That uniqueness was
not an accident -- lesson 5's tool module was named `store.py` specifically to
avoid clashing with lesson 2's `tools.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LESSONS = Path(__file__).resolve().parents[1]
REPO_ROOT = LESSONS.parent

# Order matters only in that every directory must be present before any test
# module imports from it. Lesson 3's `toolset` imports lesson 2's `tools`, and
# lesson 5 imports lesson 4's `context`, so the whole set goes on at once.
for folder in (
    "01-structured-output",
    "02-tool-calling",
    "03-agent-loop",
    "04-memory-context",
    "05-retrieval",
):
    path = str(LESSONS / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

# This lesson's own modules (fakes.py) -- needed when pytest is invoked from the
# repo root rather than from this directory.
_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers.

    `live` is the important one. Tests that call a real model are slow, cost
    tokens, and fail when a provider rate-limits you -- so they must not run by
    default. A suite you avoid running because it is expensive provides no safety
    at all.

        uv run pytest lessons/06-testing                    # offline, free, fast
        uv run pytest lessons/06-testing -m live             # only the live ones
        uv run pytest lessons/06-testing -m "not live"       # explicit default
    """
    config.addinivalue_line(
        "markers", "live: hits a real model API. Costs tokens. Deselected by default."
    )
    config.addinivalue_line(
        "markers", "cassette: replays a recorded model response from cassettes/."
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live tests unless they were explicitly asked for.

    Opt-in rather than opt-out: the default `pytest` invocation must be safe to run
    on a plane, in CI, or a hundred times an hour while refactoring.
    """
    selected_marker = config.getoption("-m", default="")
    if "live" in selected_marker:
        return

    skip_live = pytest.mark.skip(reason="live test; run with -m live to include")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def registry():
    """Lesson 3's six-tool registry: time, calculate, currency, and three fs tools."""
    from toolset import build_registry

    return build_registry()


@pytest.fixture
def live_client():
    """A real client, or skip. Only used by tests marked `live`."""
    from llmkit import ConfigError, get_client

    try:
        return get_client()
    except ConfigError as exc:
        pytest.skip(f"no usable model configured: {exc}")

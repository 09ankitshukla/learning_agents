"""Tools for lesson 3: lesson 2's three, plus three that read the filesystem.

Two things worth noticing about this file.

**Lesson 2's tools are imported, not copied.** The claim "the only difference
between lesson 2 and lesson 3 is the control flow" should be literally true, so
`get_current_time`, `calculate` and `convert_currency` are the same code running
under a loop instead of a single round. Lesson folders are not importable packages
(their names start with digits), so the path is added explicitly below. This file
is named `toolset.py` rather than `tools.py` precisely so that `import tools`
unambiguously resolves to lesson 2's module.

**The filesystem tools introduce a new class of risk.** Arithmetic on a bad
expression is contained; a path is not. `read_file("../../../../etc/passwd")` is a
perfectly well-formed tool call. So these tools are sandboxed, and the sandbox is
the interesting part -- see `_safe_path`.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

from llmkit import ToolError, ToolRegistry, ToolSpec

# --- reuse lesson 2's tools verbatim ---------------------------------------
_LESSON_02 = Path(__file__).resolve().parents[1] / "02-tool-calling"
if str(_LESSON_02) not in sys.path:
    sys.path.insert(0, str(_LESSON_02))

from tools import (  # noqa: E402  (path must be set first)
    CALCULATE_SPEC,
    CURRENCY_SPEC,
    TIME_SPEC,
    calculate,
    convert_currency,
    get_current_time,
)

# ---------------------------------------------------------------------------
# The sandbox
# ---------------------------------------------------------------------------
# Everything the agent may read lives under this directory. Note it is the repo
# itself, which makes the research task self-referential in a way that is useful
# later: lesson 5 builds retrieval over these same files.
SANDBOX = Path(__file__).resolve().parents[2]

# Containment is the primary control, and it is an allowlist: a resolved path is
# either inside SANDBOX or it is refused. This denylist is a *second*, narrower
# policy for files that are inside the sandbox but still must not be read.
#
# Keep the two ideas separate in your head. The allowlist stops path traversal.
# The denylist expresses "this specific thing is secret" -- and `.env` holds your
# API key, which an agent has no business reading and which you certainly do not
# want pasted into a prompt and sent to a model provider.
DENIED_PARTS = {".env", ".git", ".venv", "__pycache__", "node_modules"}

MAX_READ_LINES = 200
MAX_LIST_ENTRIES = 60
MAX_SEARCH_HITS = 25


def _safe_path(raw: str) -> Path:
    """Resolve a model-supplied path, or refuse it.

    The check that matters is `is_relative_to(SANDBOX)` *after* `resolve()`.
    Resolving first is essential: it collapses `..` and follows symlinks, so
    "lessons/../../../../etc/passwd" becomes an absolute path that plainly fails
    the containment test. Checking the string before resolving is the classic
    path-traversal bug -- `"../"` in a string is easy to blocklist and easy to
    smuggle past a blocklist.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ToolError("path must be a non-empty string, relative to the project root.")

    candidate = (SANDBOX / raw.strip()).resolve()

    if not candidate.is_relative_to(SANDBOX):
        raise ToolError(
            f"Path {raw!r} is outside the project directory and cannot be accessed. "
            f"Use paths relative to the project root, such as 'lessons' or 'README.md'."
        )

    if DENIED_PARTS.intersection(candidate.parts):
        raise ToolError(
            f"Path {raw!r} touches a restricted location and cannot be accessed."
        )

    return candidate


def _relative(path: Path) -> str:
    return path.relative_to(SANDBOX).as_posix()


# ---------------------------------------------------------------------------
# Tool 4: list files
# ---------------------------------------------------------------------------
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files and folders, so the agent can orient itself.

    An agent exploring an unknown filesystem needs this first -- it cannot read a
    file whose name it does not know. In practice this is the tool that makes the
    loop multi-step: list, then read, then answer.
    """
    target = _safe_path(directory)
    if not target.exists():
        raise ToolError(f"'{directory}' does not exist.")
    if not target.is_dir():
        raise ToolError(f"'{directory}' is a file, not a directory. Use read_file instead.")

    dirs: list[str] = []
    files: list[str] = []
    for entry in sorted(target.iterdir()):
        if DENIED_PARTS.intersection(entry.parts):
            continue
        if entry.is_dir():
            dirs.append(f"{entry.name}/")
        elif fnmatch.fnmatch(entry.name, pattern):
            files.append(entry.name)

    if not dirs and not files:
        return f"'{directory}' contains nothing matching {pattern!r}."

    lines = [f"Contents of '{_relative(target) or '.'}':"]
    for name in dirs[:MAX_LIST_ENTRIES]:
        lines.append(f"  {name}")
    for name in files[:MAX_LIST_ENTRIES]:
        lines.append(f"  {name}")
    total = len(dirs) + len(files)
    if total > MAX_LIST_ENTRIES:
        lines.append(f"  ... and {total - MAX_LIST_ENTRIES} more")
    return "\n".join(lines)


LIST_FILES_SPEC = ToolSpec(
    name="list_files",
    description=(
        "List the files and subdirectories inside a directory of this project. "
        "Use this first to discover what exists before trying to read anything. "
        "Directory names end with '/'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": (
                    "Directory path relative to the project root. Use '.' for the "
                    "root, or e.g. 'lessons', 'docs', 'lessons/01-structured-output'."
                ),
            },
            "pattern": {
                "type": "string",
                "description": "Optional glob filter for filenames, e.g. '*.md'.",
            },
        },
        "required": [],
    },
)


# ---------------------------------------------------------------------------
# Tool 5: read a file
# ---------------------------------------------------------------------------
def read_file(path: str, start_line: int = 1, max_lines: int = MAX_READ_LINES) -> str:
    """Read part of a text file.

    Note the truncation. A tool result goes straight into the conversation and is
    re-sent on every subsequent model call, so an unbounded read is a context-window
    bomb: one 5,000-line file can consume the entire budget and push out the
    conversation that gave the task meaning.

    Truncating and *saying so* lets the model ask for the next chunk. Truncating
    silently makes it confidently answer from half a file.
    """
    target = _safe_path(path)
    if not target.exists():
        raise ToolError(f"'{path}' does not exist. Use list_files to see what does.")
    if target.is_dir():
        raise ToolError(f"'{path}' is a directory. Use list_files instead.")

    try:
        start_line = max(1, int(start_line))
        max_lines = max(1, min(int(max_lines), MAX_READ_LINES))
    except (TypeError, ValueError) as exc:
        raise ToolError("start_line and max_lines must be integers.") from exc

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"Could not read '{path}': {exc}") from exc

    lines = content.splitlines()
    chunk = lines[start_line - 1 : start_line - 1 + max_lines]
    if not chunk:
        raise ToolError(
            f"'{path}' has only {len(lines)} lines; start_line={start_line} is past the end."
        )

    numbered = "\n".join(
        f"{i:>5} | {text}" for i, text in enumerate(chunk, start=start_line)
    )
    header = f"{_relative(target)} (lines {start_line}-{start_line + len(chunk) - 1} of {len(lines)})"
    footer = ""
    if start_line + len(chunk) - 1 < len(lines):
        footer = (
            f"\n\n[truncated: {len(lines) - (start_line + len(chunk) - 1)} more lines. "
            f"Call read_file again with start_line={start_line + len(chunk)} to continue.]"
        )
    return f"{header}\n{numbered}{footer}"


READ_FILE_SPEC = ToolSpec(
    name="read_file",
    description=(
        "Read the contents of a text file in this project, with line numbers. "
        "Long files are truncated; the result tells you how to fetch the next part."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the project root, e.g. 'docs/glossary.md'.",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read, 1-based. Defaults to 1.",
            },
            "max_lines": {
                "type": "integer",
                "description": f"How many lines to read, up to {MAX_READ_LINES}.",
            },
        },
        "required": ["path"],
    },
)


# ---------------------------------------------------------------------------
# Tool 6: search
# ---------------------------------------------------------------------------
def search_files(query: str, directory: str = ".", file_pattern: str = "*.md") -> str:
    """Find which files mention a string, and where.

    This is the tool that makes multi-step behaviour obvious: search returns
    locations, not answers, so the model has to read something afterwards. It is
    also a deliberate preview of lesson 5 -- this is keyword search, and its
    limitation (you must guess the exact word) is what motivates embeddings.
    """
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a non-empty string.")

    root = _safe_path(directory)
    if not root.is_dir():
        raise ToolError(f"'{directory}' is not a directory.")

    needle = query.strip().lower()
    hits: list[str] = []
    files_scanned = 0

    for path in sorted(root.rglob("*")):
        if len(hits) >= MAX_SEARCH_HITS:
            break
        if not path.is_file() or DENIED_PARTS.intersection(path.parts):
            continue
        if not fnmatch.fnmatch(path.name, file_pattern):
            continue
        files_scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                snippet = line.strip()[:160]
                hits.append(f"{_relative(path)}:{number}: {snippet}")
                if len(hits) >= MAX_SEARCH_HITS:
                    break

    if not hits:
        return (
            f"No matches for {query!r} in {files_scanned} file(s) matching "
            f"{file_pattern!r} under '{directory}'. Try a different word, or a "
            f"broader file_pattern such as '*'."
        )

    capped = " (capped)" if len(hits) >= MAX_SEARCH_HITS else ""
    return (
        f"{len(hits)} match(es){capped} for {query!r} in {files_scanned} file(s):\n"
        + "\n".join(hits)
    )


SEARCH_FILES_SPEC = ToolSpec(
    name="search_files",
    description=(
        "Search the project's text files for a word or phrase and return the "
        "matching file paths with line numbers and the matching line. Use this to "
        "locate where a topic is discussed, then read_file to see the context. "
        "Matching is case-insensitive and literal, not semantic -- you must guess "
        "the actual wording used."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Literal text to search for."},
            "directory": {
                "type": "string",
                "description": "Directory to search under, relative to the project root. Defaults to '.'.",
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob for filenames to search, e.g. '*.md' (default) or '*.py'.",
            },
        },
        "required": ["query"],
    },
)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def build_registry() -> ToolRegistry:
    """All six tools. Note how little wiring a tool needs: a function and a spec."""
    registry = ToolRegistry()
    registry.add(TIME_SPEC, get_current_time)
    registry.add(CALCULATE_SPEC, calculate)
    registry.add(CURRENCY_SPEC, convert_currency)
    registry.add(LIST_FILES_SPEC, list_files)
    registry.add(READ_FILE_SPEC, read_file)
    registry.add(SEARCH_FILES_SPEC, search_files)
    return registry

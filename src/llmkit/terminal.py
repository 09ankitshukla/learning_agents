"""A terminal that can print what a model actually produces.

Why this module exists, because it is not obvious and it cost a real debugging
session:

Windows consoles default to the cp1252 code page. Language models cheerfully
emit non-breaking hyphens, typographic quotes, em dashes, arrows and emoji --
none of which exist in cp1252. Printing one raises:

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2011'

The failure has nothing to do with your agent logic, it happens at print time
rather than at generation time, and it takes down the whole script. Milder cases
just mangle output, which is how "the table's size" renders as "the tableÆs
size".

Reconfiguring the streams to UTF-8 with errors="replace" makes output correct
where the terminal supports it and merely imperfect where it does not. Never a
crash.

Every lesson imports `console` from here so the fix applies once.
"""

from __future__ import annotations

import sys

from rich.console import Console


def _force_utf8() -> None:
    """Make Python emit UTF-8, and make the Windows console read it as UTF-8.

    Both halves are needed. Reconfiguring the streams alone stops the crash but
    leaves the console interpreting UTF-8 bytes as cp437, which turns box-drawing
    characters into mojibake. Setting the console code page to 65001 fixes the
    reading side.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:  # noqa: BLE001 - cosmetic; never worth failing over
            pass

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            # Not a real text stream (captured output, a pipe, a test harness).
            pass


_force_utf8()

#: Shared console for all lessons. Import this rather than constructing your own.
console = Console()

"""Scout: a purpose-aware company intelligence agent."""

import sys

if sys.version_info < (3, 11):  # noqa: UP036 - friendly error for older interpreters
    raise RuntimeError(
        f"Scout needs Python 3.11 or newer; this is {sys.version.split()[0]}. See the README."
    )

__version__ = "0.1.0"

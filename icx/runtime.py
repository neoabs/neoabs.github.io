"""Checks shared by the notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

from icx.config import find_root


def require_project_venv(start: Path | None = None) -> Path:
    """Return the project root, or raise if this kernel is not ``.venv``."""
    root = find_root(start)
    expected = (root / ".venv").resolve()
    current = Path(sys.prefix).resolve()
    if current != expected:
        raise RuntimeError(
            "Run this notebook with the project .venv.\n"
            f"Expected: {expected / 'bin' / 'python'}\n"
            f"Current:  {sys.executable}"
        )
    return root

from __future__ import annotations

import os
from pathlib import Path


def locate_xfoil(root: Path) -> Path | None:
    configured = os.getenv("XFOIL_EXE", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates += [root / "tools" / "xfoil" / "xfoil.exe", root / "tools" / "xfoil" / "xfoil"]
    return next((path.resolve() for path in candidates if path.is_file()), None)

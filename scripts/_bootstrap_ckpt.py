"""Set ODE_CKPT before importing app (loads model at import time)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_ckpt() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/arcus_ode_lab/.matplotlib")
    if not os.environ.get("ODE_CKPT", "").strip():
        for candidate in (
            root / "model" / "ode.pt",
            Path.cwd() / "model" / "ode.pt",
            root / "ode.pt",
            Path.cwd() / "ode.pt",
            Path.home() / "Downloads" / "ode.pt",
        ):
            if candidate.is_file():
                os.environ["ODE_CKPT"] = str(candidate.resolve())
                break
    return Path(os.environ.get("ODE_CKPT", root / "model" / "ode.pt")).expanduser()
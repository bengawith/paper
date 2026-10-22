from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def configure_phase_logging(report_root: Path, phase_id: str) -> tuple[logging.Logger, Path]:
    log_dir = report_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    readable = log_dir / f"{phase_id}.log"
    logger = logging.getLogger(f"robust_airfoil.{phase_id}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(readable, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, readable


def append_jsonl(report_root: Path, phase_id: str, event: str, **payload: object) -> None:
    path = report_root / "logs" / f"{phase_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")

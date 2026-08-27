from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from robust_airfoil.hashing import hash_paths
from robust_airfoil.provenance import utc_now, write_json_atomic

PhaseStatus = Literal["pending", "running", "passed", "held", "failed"]


@dataclass
class PhaseRecord:
    phase_id: str
    name: str
    status: PhaseStatus = "pending"
    started_utc: str | None = None
    finished_utc: str | None = None
    git_sha: str = ""
    config_hashes: dict[str, str] = field(default_factory=dict)
    input_hashes: dict[str, str] = field(default_factory=dict)
    output_hashes: dict[str, str] = field(default_factory=dict)
    command: str = ""
    log_path: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    failure_type: str | None = None
    failure_message: str | None = None
    retryable: bool = False


class RunState:
    def __init__(self, path: Path):
        self.path = path
        self.phases: dict[str, PhaseRecord] = {}
        if path.exists():
            import json

            raw = json.loads(path.read_text(encoding="utf-8"))
            self.phases = {key: PhaseRecord(**value) for key, value in raw.get("phases", {}).items()}

    def save(self) -> None:
        write_json_atomic(self.path, {"schema_version": 1, "phases": {k: asdict(v) for k, v in self.phases.items()}})

    def begin(self, phase_id: str, name: str, **metadata: Any) -> PhaseRecord:
        record = PhaseRecord(phase_id=phase_id, name=name, status="running", started_utc=utc_now(), **metadata)
        self.phases[phase_id] = record
        self.save()
        return record

    def finish(self, phase_id: str, status: PhaseStatus, summary: dict[str, Any], outputs: list[Path] | None = None) -> None:
        record = self.phases[phase_id]
        record.status = status
        record.finished_utc = utc_now()
        record.summary = summary
        record.output_hashes = hash_paths(outputs or []) if outputs else {}
        self.save()

    def fail(self, phase_id: str, failure_type: str, message: str, retryable: bool) -> None:
        record = self.phases[phase_id]
        record.status = "failed"
        record.finished_utc = utc_now()
        record.failure_type = failure_type
        record.failure_message = message
        record.retryable = retryable
        self.save()

    def can_skip(self, phase_id: str, config_hashes: dict[str, str], input_hashes: dict[str, str]) -> bool:
        record = self.phases.get(phase_id)
        if not record or record.status not in {"passed", "held"}:
            return False
        if record.config_hashes != config_hashes or record.input_hashes != input_hashes:
            return False
        for path_text, expected in record.output_hashes.items():
            path = Path(path_text)
            current = hash_paths([path]) if path.exists() else {}
            if not current or next(iter(current.values())) != expected:
                return False
        return True

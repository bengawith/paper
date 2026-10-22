import json

import pytest

from robust_airfoil.reports.viability import REQUIRED_FIELDS, build_viability_report


def test_report_requires_every_contract_field(tmp_path):
    with pytest.raises(ValueError):
        build_viability_report({}, tmp_path / "report.md", tmp_path / "report.json")
    payload = {field: "evidence" for field in REQUIRED_FIELDS}
    build_viability_report(payload, tmp_path / "report.md", tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == payload
    assert "## Exact resume command" in (tmp_path / "report.md").read_text(encoding="utf-8")

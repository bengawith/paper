from robust_airfoil.run_state import RunState


def test_run_state_roundtrip_and_skip(tmp_path):
    output = tmp_path / "artifact.txt"
    output.write_text("evidence", encoding="utf-8")
    state = RunState(tmp_path / "run_state.json")
    state.begin("01", "test", config_hashes={"config": "x"}, input_hashes={"input": "y"})
    state.finish("01", "passed", {"ok": True}, [output])
    reloaded = RunState(state.path)
    assert reloaded.phases["01"].status == "passed"
    assert reloaded.can_skip("01", {"config": "x"}, {"input": "y"})
    output.write_text("changed", encoding="utf-8")
    assert not reloaded.can_skip("01", {"config": "x"}, {"input": "y"})


def test_held_phase_does_not_silently_skip_on_resume(tmp_path):
    output = tmp_path / "held-artifact.txt"
    output.write_text("complete hold evidence", encoding="utf-8")
    state = RunState(tmp_path / "run_state.json")
    state.begin("22", "report", config_hashes={"config": "x"}, input_hashes={})
    state.finish("22", "held", {"decision": "HOLD"}, [output])
    reloaded = RunState(state.path)
    assert not reloaded.can_skip("22", {"config": "x"}, {})

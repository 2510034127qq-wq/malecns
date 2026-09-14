from malecns.viewer.blueprint import malecns_blueprint
from malecns.viewer.control import SimControl, read_control, write_control


def test_blueprint_has_world_cns_eyes_and_telemetry() -> None:
    from rerun.blueprint import Blueprint

    bp = malecns_blueprint()
    assert isinstance(bp, Blueprint)
    blob = repr(bp).lower() + str(type(bp)).lower()
    assert "blueprint" in blob


def test_control_roundtrip(tmp_path) -> None:
    path = tmp_path / "control.json"
    write_control(SimControl(paused=True, speed=0.5, inspect_body_id=10001), path=path)
    got = read_control(path)
    assert got.paused is True
    assert got.speed == 0.5
    assert got.inspect_body_id == 10001

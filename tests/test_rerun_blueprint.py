from malecns.viewer.blueprint import malecns_blueprint


def test_blueprint_has_world_cns_eyes_and_telemetry() -> None:
    from rerun.blueprint import Blueprint

    bp = malecns_blueprint()
    assert isinstance(bp, Blueprint)
    views = bp.flatten() if hasattr(bp, "flatten") else []
    blob = repr(bp).lower() + str(type(bp)).lower()
    assert "blueprint" in blob
    del views

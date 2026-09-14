from malecns.cli import main


def test_validate_env_cli_json(capsys) -> None:
    assert main(["validate-env", "--allow-cpu"]) == 0
    out = capsys.readouterr().out
    assert '"arbor_version": "0.12.2"' in out
    assert '"rerun_available": true' in out

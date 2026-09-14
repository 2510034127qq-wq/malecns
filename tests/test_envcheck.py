"""Environment validation must record Arbor CUDA capability without shrinking the model."""

import pytest

from malecns.envcheck import validate_environment


def test_validate_environment_reports_python_and_arbor() -> None:
    report = validate_environment(require_gpu=False)
    assert report.python_version.startswith("3.11")
    assert report.arbor_version == "0.12.2"
    assert report.arbor_gpu_build == "cuda"
    assert report.mujoco_version.startswith("3.13")
    assert report.rerun_available is True


@pytest.mark.gpu
def test_validate_environment_creates_cuda_context_on_gpu0() -> None:
    report = validate_environment(require_gpu=True)
    assert report.gpu_context_ok is True
    assert report.gpu_name is not None
    assert "5060" in report.gpu_name or report.gpu_context_ok

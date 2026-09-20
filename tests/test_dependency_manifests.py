"""Guard against distributions that overwrite the same installed modules."""

from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
OPENCV = {
    "opencv-python", "opencv-contrib-python",
    "opencv-python-headless", "opencv-contrib-python-headless",
}
ORT = {"onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"}


def requirements(filename):
    parsed = {}
    for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            item = Requirement(line)
            assert item.name not in parsed, f"Duplicate dependency: {item.name}"
            parsed[item.name] = item
    return parsed


def test_base_worker_does_not_install_overlapping_runtime_distributions():
    deps = requirements("requirements.txt")
    assert OPENCV.intersection(deps) == {"opencv-python"}
    assert not ORT.intersection(deps)
    # This upstream package pulls in a second OpenCV and CPU ONNX Runtime.
    assert "rtmlib" not in deps
    assert deps["torch"].specifier.contains("2.11.0+cu128")
    assert deps["torchvision"].specifier.contains("0.26.0+cu128")
    assert not deps["setuptools"].specifier.contains("84.0.0")


def test_web_profile_contains_runtime_server_oauth_and_billing_dependencies():
    deps = requirements("requirements-web.txt")
    assert {"httpx", "authlib", "stripe", "uvicorn"} <= deps.keys()
    assert OPENCV.intersection(deps) == {"opencv-python-headless"}
    assert not {"torch", "ultralytics", "sam2", "rtmlib"}.intersection(deps)


def test_each_optional_pose_profile_supplies_one_runtime_and_actual_dependencies():
    for filename, backend in (
        ("requirements-rtm-cpu.txt", "onnxruntime"),
        ("requirements-rtm-cuda12.txt", "onnxruntime-gpu"),
    ):
        deps = requirements(filename)
        assert {"numpy", "opencv-python", "tqdm", backend} <= deps.keys()
        assert OPENCV.intersection(deps) == {"opencv-python"}
        assert ORT.intersection(deps) == {backend}
        assert deps[backend].specifier.contains("1.26.0")

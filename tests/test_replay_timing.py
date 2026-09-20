import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_browser_skeleton_timing_does_not_show_future_or_stale_poses():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to exercise the actual browser timing functions")
    templates = Path(__file__).resolve().parents[1] / "app" / "templates"
    replay = (templates / "replay.html").read_text(encoding="utf-8")
    live = (templates / "progress.html").read_text(encoding="utf-8")
    functions = [line for line in replay.splitlines() if line.startswith(("function surrounding(", "function poseAt("))]
    functions += [line for line in live.splitlines() if line.startswith("function freshObservation(")]
    script = "\n".join(functions) + """
let frames=[],video={currentTime:0};
const frame=(t,x)=>({time_seconds:t,fighter_A:{identity_confidence:.9,observation:{keypoints:[[x,x]]}}});
frames=[frame(2,10),frame(2.1,30)];
const before=poseAt('fighter_A',1), middle=poseAt('fighter_A',2.05), after=poseAt('fighter_A',3);
frames=[frame(2,10),frame(3,30)];
const gap=poseAt('fighter_A',2.95);
video.currentTime=.95;const future=freshObservation({time_seconds:1});
video.currentTime=1.1;const current=freshObservation({time_seconds:1});
video.currentTime=1.3;const old=freshObservation({time_seconds:1});
process.stdout.write(JSON.stringify({before,middle,after,gap,future,current,old}));
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    values = json.loads(result.stdout)
    assert values["before"] is None
    assert values["after"] is None
    assert values["gap"] is None
    assert values["middle"][0] == pytest.approx([20, 20])
    assert values["future"] is False
    assert values["current"] is True
    assert values["old"] is False

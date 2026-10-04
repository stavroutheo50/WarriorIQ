"""The pre-deploy rules (core/predeploy_checks.py), on hand-made reports.

tools/predeploy_regression.py runs the same checks on real clips. Dry-run on
2026-10-04 with synthetic stand-ins on CPU (fight, waist-up, solo, sideways,
no people, corrupt): every rule passed except timing, which needs the GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import predeploy_checks as checks

ROOT = Path(__file__).resolve().parents[1]


def _span(start, end, duration):
    return {"video": {"analysed_span": {"start_seconds": start, "end_seconds": end,
                                        "video_duration_seconds": duration}}}


def test_whole_video():
    assert checks.whole_video(_span(0.0, 30.0, 30.0)) == []
    assert "started at 18.0s" in checks.whole_video(_span(18.0, 30.0, 30.0))[0]
    assert checks.whole_video({}) != []


def test_timing():
    assert checks.within_video_length(29.0, _span(0, 30, 30)) == []
    assert "took 31.0s" in checks.within_video_length(31.0, _span(0, 30, 30))[0]


def _kick(conf):
    kp = np.zeros((17, 2)) + 50
    c = np.ones(17)
    c[13:17] = conf
    sample = {"attacker_keypoints": kp.tolist(), "attacker_conf": c.tolist()}
    return {"family": "kick", "limb": "right_leg", "peak_time": 4.0,
            "evidence": {"contact_samples": [sample, sample]}}


def test_no_kick_without_legs():
    assert checks.no_kicks_without_legs([_kick(0.9)]) == []
    assert checks.no_kicks_without_legs([_kick(0.1)]) != []
    assert checks.no_kicks_without_legs([_kick(0.9)], legs_in_shot=False) != []
    assert checks.no_kicks_without_legs([{"family": "punch"}], legs_in_shot=False) == []


def test_one_verdict_catches_a_plan_without_identity():
    report = json.loads((ROOT / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
    assert checks.one_verdict(report) == []
    report["tracking"]["fighter_B_coverage"] = 0.1
    assert checks.one_verdict(report) == []      # refresh withholds the plan itself
    assert checks.one_verdict({"mode": "solo", "scorecard": {"available": True}}) == ["a solo session was scored"]


def test_the_manifest_names_every_clip_the_suite_needs():
    manifest = json.loads((ROOT / "dataset" / "predeploy" / "manifest.json").read_text(encoding="utf-8"))
    kinds = [clip["kind"] for clip in manifest["clips"]]
    rulesets = {clip.get("ruleset") for clip in manifest["clips"] if clip["kind"] == "fight"}
    assert {"BOXING", "MUAY_THAI", "WT_TAEKWONDO", "MMA", "K1"} <= rulesets
    for kind in ("solo", "waist_up", "rotated", "no_people", "corrupt"):
        assert kind in kinds

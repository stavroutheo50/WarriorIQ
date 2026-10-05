from copy import deepcopy

import pytest

from core.annotations import accuracy_summary
from core.regression_manifest import build_regression_manifest, validate_regression_manifest
from core.release_validation import end_to_end_metadata


def annotation(**corrected):
    labels = {"fighter": "A", "technique": "jab", "family": "punch",
              "limb": "left_hand", "target": "head", "outcome": "clean"}
    return {"job_id": "run-1", "event_time": 12.0, "ruleset": "K1",
            "predicted": deepcopy(labels), "corrected": {**labels, **corrected}}


def test_missing_contact_timestamp_is_not_perfect_timing():
    summary = accuracy_summary([annotation()])
    assert summary["timing"]["samples"] == 0
    assert summary["timing"]["mean_absolute_error_seconds"] is None


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1])
def test_invalid_contact_timestamp_is_not_measured(value):
    assert accuracy_summary([annotation(contact_time=value)])["timing"]["samples"] == 0


def test_missing_truth_is_not_a_correct_prediction_or_positive_action():
    item = annotation()
    item["predicted"] = {}
    item["corrected"] = {}
    summary = accuracy_summary([item])
    assert all(metric["accuracy"] is None for metric in summary["metrics"].values())
    assert summary["positive_labels"] == summary["negative_labels"] == 0
    assert summary["technique_validation"]["samples"] == 0


def test_negative_label_cannot_validate_identity_target_or_outcome():
    item = annotation(technique="none", family="none", limb="none", target=None, outcome="uncertain")
    summary = accuracy_summary([item])
    assert summary["negative_labels"] == 1
    for name in ("fighter_identity", "target", "outcome", "limb_side", "legality"):
        assert summary["metrics"][name]["total"] == 0
    assert summary["technique_validation"]["false_alarms"] == 1


def test_unreviewed_fields_are_excluded_but_real_errors_still_count():
    summary = accuracy_summary([annotation(fighter="B", limb="hand", target=None, outcome="uncertain")])
    assert summary["metrics"]["fighter_identity"]["accuracy"] == 0
    for name in ("limb_side", "target", "outcome"):
        assert summary["metrics"][name]["accuracy"] is None


def test_unknown_and_coarse_techniques_do_not_become_none_training_labels():
    items = [annotation(technique=value) for value in ("kick", "typo", None)]
    summary = accuracy_summary(items)
    assert summary["technique_validation"]["samples"] == 0
    assert summary["negative_labels"] == 0


def test_two_reports_of_same_video_are_one_source_fight():
    fights = [{"fight_id": job, "video_sha256": "a" * 64, "report_sha256": "b" * 64,
               "annotations": [annotation(contact_time=12.1)]} for job in ("run-1", "run-2")]
    rows = validate_regression_manifest(build_regression_manifest(fights, created_at="test"))
    summary = accuracy_summary(rows)
    assert summary["fights"] == 1
    assert end_to_end_metadata(summary)["fights"] == 1


def test_unidentified_source_jobs_do_not_count_toward_release_fight_gate():
    summary = accuracy_summary([annotation(contact_time=12.1)])
    assert end_to_end_metadata(summary)["fights"] == 0


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -1])
def test_manifest_rejects_invalid_timestamps(timestamp):
    item = annotation()
    item["event_time"] = timestamp
    with pytest.raises(RuntimeError, match="event time"):
        build_regression_manifest([
            {"fight_id": "run-1", "video_sha256": "a" * 64, "annotations": [item]}
        ], created_at="test")


def test_report_does_not_claim_full_fight_accuracy_from_candidate_reviews(tmp_path):
    from tools.report_accuracy_benchmark import build_report, markdown_report

    manifest = build_regression_manifest([
        {"fight_id": "run-1", "video_sha256": "a" * 64,
         "annotations": [annotation(contact_time=12.1)]}
    ], created_at="test")
    report = build_report(manifest, tmp_path / "development", tmp_path / "test")
    assert report["full_fight_recall"] is None
    assert report["identity_swap_rate"] is None
    assert not report["ready_for_training"]
    assert not report["ready_for_release"]
    assert report["reviewed_candidates"]["metrics"]["fighter_identity"]["accuracy"] == 1
    assert "not product accuracy claims" in markdown_report(report)


@pytest.mark.parametrize("output_name", ["report.json", "manifest.json", "manifest.md", "report.md"])
def test_report_cli_writes_readable_and_machine_reports_without_changing_inputs(tmp_path, output_name):
    import json
    import subprocess
    import sys
    from pathlib import Path

    manifest = build_regression_manifest([
        {"fight_id": "run-1", "video_sha256": "a" * 64, "annotations": [annotation()]}
    ], created_at="test")
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    original = source.read_bytes()
    destination = tmp_path / output_name
    result = subprocess.run([
        sys.executable, "tools/report_accuracy_benchmark.py", "--manifest", str(source),
        "--development", str(tmp_path / "dev"), "--untouched-test", str(tmp_path / "test"),
        "--output", str(destination),
    ], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert source.read_bytes() == original
    if output_name != "report.json":
        assert result.returncode != 0
        return
    assert result.returncode == 0, result.stderr
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["reviewed_candidates"]["timing"]["samples"] == 0
    assert len(report["manifest_file_sha256"]) == 64
    assert destination.with_suffix(".md").is_file()

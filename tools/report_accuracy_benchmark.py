"""Report existing reviewed evidence and data gaps; never train or promote a model."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.annotations import accuracy_summary
from core.model_validation import audit_dataset_split
from core.regression_manifest import file_sha256, validate_regression_manifest
from core.release_validation import assess_end_to_end_validation, end_to_end_metadata


def build_report(manifest: dict, development: Path, untouched_test: Path) -> dict:
    annotations = validate_regression_manifest(manifest)
    summary = accuracy_summary(annotations)
    audit = audit_dataset_split(development, untouched_test)
    for name in ("development", "untouched_test"):
        audit[name].pop("sequence_fingerprints", None)
    gate = assess_end_to_end_validation(end_to_end_metadata(summary))
    return {
        "schema": "warrioriq.accuracy_benchmark.v1",
        "scope": "Archived reviewed candidates, not a fresh full-fight accuracy measurement",
        "manifest_created_at": manifest.get("created_at"),
        "manifest_content_sha256": manifest["content_sha256"],
        "analysis_runs": manifest["fight_count"],
        "reviewed_candidates": summary,
        "dataset": audit,
        "end_to_end_gate": gate,
        # Selected candidate reviews have no denominator for all real strikes
        # or tracked frames, even if their conditional accuracy looks perfect.
        "full_fight_recall": None,
        "identity_swap_rate": None,
        "ready_for_training": audit["development"]["experimental_train_ready"],
        "ready_for_release": False,
        "limitations": [
            "Predictions are from archived reports, not necessarily the current engine.",
            "Candidate reviews cannot measure full-fight missed-strike rate or identity swaps.",
            "Matching file hashes identify exact duplicate videos, not re-encodes of the same bout.",
            "Legacy review provenance and complete-interval coverage still need independent confirmation.",
            "AI-generated labels and automatic negatives are not independent ground truth.",
        ],
    }


def markdown_report(report: dict) -> str:
    summary = report["reviewed_candidates"]
    dev = report["dataset"]["development"]
    test = report["dataset"]["untouched_test"]
    lines = [
        "# Warrior IQ accuracy benchmark", "", report["scope"], "",
        f"- Archived analysis runs: {report['analysis_runs']}",
        f"- Distinct source-video hashes: {summary['verified_source_fights']}",
        f"- Reviewed candidates: {summary['annotations']}",
        "- Full-fight recall and identity-swap rate: not measured", "",
        "## Reviewed candidate measurements", "",
        "These small-sample results are diagnostic, not product accuracy claims.", "",
        "| Field | Correct / reviewed | Accuracy |", "| --- | ---: | ---: |",
    ]
    for name, item in summary["metrics"].items():
        accuracy = "Not measured" if item["accuracy"] is None else f"{item['accuracy']:.1%}"
        lines.append(f"| {name} | {item['correct']} / {item['total']} | {accuracy} |")
    timing = summary["timing"]
    mae = timing["mean_absolute_error_seconds"]
    classification = summary["technique_validation"]
    lines.extend([
        "", f"Classifiable candidate pairs: {classification['samples']}; "
        f"false alarms: {classification['false_alarms']}; "
        f"reviewed missed actions: {classification['missed_actions']}.",
        "These counts exclude unlabelled pairs and do not measure missed actions across the whole fight.",
        "", f"Explicit contact timestamps: {timing['samples']}. " +
        ("Timing error is not measured." if mae is None else f"Mean absolute error: {mae:.3f}s."),
        "", "## Training data", "",
        f"- {dev['valid_sequences']} valid sequences: {dev['positive_sequences']} positive, {dev['negative_sequences']} negative.",
        f"- {dev['duplicate_sequences']} duplicate sequences; originals have not been changed.",
        f"- {dev['covered_classes']} / {dev['total_classes']} classes represented.",
        f"- Missing classes: {', '.join(dev['missing_classes']) or 'none'}.",
        f"- Untouched test sequences: {test['valid_sequences']}.",
        "", "## Release checks", "",
    ])
    lines.extend(f"- {failure}" for failure in report["end_to_end_gate"]["failures"])
    lines.extend(["", "No model was trained, promoted or deployed by this report.", "",
                  "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("dataset/regression/private/manifest.json"))
    parser.add_argument("--development", type=Path, default=Path("dataset/sequences"))
    parser.add_argument("--untouched-test", type=Path, default=Path("dataset/untouched_test"))
    parser.add_argument("--output", type=Path, default=Path("dataset/regression/private/accuracy-benchmark.json"))
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("Output must be a .json path; a separate .md report is also written")
    if args.manifest.resolve() in {args.output.resolve(), args.output.with_suffix(".md").resolve()}:
        parser.error("Output must not overwrite the input manifest")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = build_report(manifest, args.development, args.untouched_test)
    report["manifest_file_sha256"] = file_sha256(args.manifest)
    # Identify the evaluator, not the archived prediction/model version.
    report["evaluator_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    report["evaluator_source_sha256"] = {
        name: file_sha256(PROJECT_ROOT / name) for name in (
            "core/annotations.py", "core/model_validation.py", "core/regression_manifest.py",
            "core/release_validation.py", "tools/report_accuracy_benchmark.py")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(f"Benchmark report: {args.output.with_suffix('.md').resolve()}")
    print(f"Training data ready: {report['ready_for_training']}; release validated: False")


if __name__ == "__main__":
    main()

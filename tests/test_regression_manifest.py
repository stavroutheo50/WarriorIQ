from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.regression_manifest import build_regression_manifest, file_sha256, validate_regression_manifest


class RegressionManifestTests(unittest.TestCase):
    def _manifest(self):
        return build_regression_manifest([
            {
                "fight_id": "fight-001",
                "video_sha256": "a" * 64,
                "report_sha256": "b" * 64,
                "annotations": [{
                    "event_time": 2.5,
                    "ruleset": "K1",
                    "predicted": {"fighter": "A", "technique": "right_jab"},
                    "corrected": {"fighter": "A", "technique": "right_jab"},
                }],
            }
        ], created_at="2026-08-30T00:00:00+00:00")

    def test_manifest_flattens_verified_annotations(self):
        annotations = validate_regression_manifest(self._manifest())
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["job_id"], "fight-001")
        self.assertEqual(annotations[0]["event_time"], 2.5)

    def test_manifest_rejects_tampered_labels(self):
        manifest = self._manifest()
        manifest["fights"][0]["annotations"][0]["corrected"]["fighter"] = "B"
        with self.assertRaisesRegex(RuntimeError, "content hash"):
            validate_regression_manifest(manifest)

    def test_file_hash_tracks_exact_asset_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            asset = Path(temporary) / "asset.bin"
            asset.write_bytes(b"real fight bytes")
            first = file_sha256(asset)
            asset.write_bytes(b"changed fight bytes")
            second = file_sha256(asset)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, second)

    def test_manifest_round_trip_is_valid_json(self):
        payload = json.loads(json.dumps(self._manifest()))
        self.assertEqual(validate_regression_manifest(payload)[0]["ruleset"], "K1")


if __name__ == "__main__":
    unittest.main()

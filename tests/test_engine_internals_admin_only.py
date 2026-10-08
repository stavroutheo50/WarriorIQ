"""Engine internals are for admins; reports name the build that made them.

QA, 2026-10-07: a fighter's report showed "OpenAI ambiguity referee", the
pose model's file name, the graphics card, "Speed budget", "Quality mode",
"Classifier: multi_frame_temporal_rules" and "build unknown". The technical
panel was gated only on the paid report tier, and the build stamp read only
git or WARRIORIQ_BUILD_COMMIT, while the cPanel deploy records its commit in
DEPLOYED_COMMIT and Render in RENDER_GIT_COMMIT.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tests_support
from tests_support import _Stub, render_result

from core import build_info

INTERNALS = ("OpenAI ambiguity referee", "Pose detector and ReID tracker", "Speed budget", "Quality mode",
             "Classifier:", "SAM2 continuous mask tracking", 'id="report-frame-breakdown"')


def _page(admin: bool) -> str:
    stub = _Stub(url=_Stub(path="/"), state=_Stub(account=None, is_admin=admin), cookies={}, headers={})
    report_perf = {"frame_pass_seconds_per_frame": 0.05, "frame_pass_breakdown_per_frame": {"pose_model": 0.03},
                   "analysis_seconds": 60.0, "segment_duration_seconds": 60.0}
    with patch.object(tests_support, "request_stub", lambda account=None: stub):
        page = render_result(analysis_build={"analysis_version": build_info.ANALYSIS_VERSION,
                                             "commit": "abc123def456", "current_version": build_info.ANALYSIS_VERSION,
                                             "outdated": False})
        report = json.loads((Path(tests_support.__file__).parent / "fixtures" / "report_sample.json")
                            .read_text(encoding="utf-8"))
        report.setdefault("performance", {}).update(report_perf)
        timed = render_result(report=report, analysis_build={"analysis_version": build_info.ANALYSIS_VERSION,
                                                             "commit": "abc123def456", "outdated": False})
    return page + timed


class AdminOnlyTests(unittest.TestCase):
    def test_a_fighter_never_sees_engine_internals(self):
        page = _page(admin=False)
        for text in INTERNALS:
            self.assertNotIn(text, page, text)
        self.assertNotIn("abc123def456", page)
        self.assertNotIn("build unknown", page)
        # What a fighter can act on stays.
        self.assertIn("first-frame identity gate", page)
        self.assertIn(f"Analysis engine v{build_info.ANALYSIS_VERSION}", page)

    def test_an_admin_still_has_them(self):
        page = _page(admin=True)
        for text in INTERNALS:
            self.assertIn(text, page, text)
        self.assertIn("Build abc123def456", page)


class BuildCommitTests(unittest.TestCase):
    def setUp(self):
        build_info.build_commit.cache_clear()
        self.addCleanup(build_info.build_commit.cache_clear)

    def _commit(self, env: dict, deployed: str | None, tmp) -> str:
        if deployed is not None:
            (tmp / "DEPLOYED_COMMIT").write_text(deployed + "\n", encoding="utf-8")
        clean = {k: v for k, v in os.environ.items() if k not in {"WARRIORIQ_BUILD_COMMIT", "RENDER_GIT_COMMIT"}}
        with patch.dict(os.environ, {**clean, **env}, clear=True), patch.object(build_info, "ROOT", tmp):
            build_info.build_commit.cache_clear()
            return build_info.build_commit()

    def test_deploy_records_are_read_before_git(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)  # no .git here, as on the cPanel host
            self.assertEqual(self._commit({}, "0f1e2d3c4b5a69788", tmp), "0f1e2d3c4b5a")
            self.assertEqual(self._commit({"RENDER_GIT_COMMIT": "aaaabbbbccccdddd"}, None, tmp), "aaaabbbbcccc")
            self.assertEqual(self._commit({"WARRIORIQ_BUILD_COMMIT": "1234567890abcdef",
                                           "RENDER_GIT_COMMIT": "aaaabbbbccccdddd"}, None, tmp), "1234567890ab")

    def test_without_any_record_it_still_says_unknown(self):
        with tempfile.TemporaryDirectory() as name:
            self.assertEqual(self._commit({}, None, Path(name)), "unknown")


if __name__ == "__main__":
    unittest.main()

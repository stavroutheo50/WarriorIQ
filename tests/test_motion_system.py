from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]


class WarriorIQMotionSystemTests(TestCase):
    def test_motion_runtime_exposes_truthful_dynamic_counting_and_primary_actions(self):
        script = (ROOT / "app" / "static" / "motion.js").read_text(encoding="utf-8")

        self.assertIn("countTo", script)
        self.assertIn("data-motion-primary", script)
        self.assertIn("prefers-reduced-motion: reduce", script)
        self.assertIn("pointer: fine", script)

    def test_home_marks_only_deliberate_motion_surfaces(self):
        template = (ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")

        self.assertIn('data-motion-sequence="steps"', template)
        self.assertIn("data-motion-primary", template)
        self.assertNotIn("DarkVeil", template)
        self.assertNotIn("DotGrid", template)

    def test_fighter_selection_has_scoped_reticle_and_lock_feedback(self):
        template = (ROOT / "app" / "templates" / "select.html").read_text(encoding="utf-8")

        self.assertIn("drawReticle", template)
        self.assertIn("drawLockCorners", template)
        self.assertIn("onpointerdown", template)
        self.assertNotIn("canvas.onmousedown", template)

    def test_live_statistics_use_the_shared_counting_api(self):
        template = (ROOT / "app" / "templates" / "progress.html").read_text(encoding="utf-8")

        self.assertIn("WarriorIQMotion?.countTo", template)
        self.assertIn("eventElementById", template)

    def test_motion_css_keeps_effects_scoped_and_reduced_motion_safe(self):
        css = (ROOT / "app" / "static" / "motion.css").read_text(encoding="utf-8")

        self.assertIn(".wiq-primary-motion", css)
        self.assertIn(".selection-lock-flash", css)
        self.assertIn("@media(prefers-reduced-motion:reduce)", css)


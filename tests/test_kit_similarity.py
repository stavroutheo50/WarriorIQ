"""Kit similarity on synthetic crops: different colours low, identical high.

QA 2026-09: yellow/red against blue/blue scored "89% alike", black against
blue/purple "79%", identical white doboks passed, and a black-and-white bout
read 100% with the report blaming the camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import kit
from core.config import SETTINGS

BGR = {
    "yellow": (0, 220, 240), "red": (30, 30, 220), "blue": (210, 70, 30), "purple": (150, 40, 110),
    "black": (20, 20, 20), "white": (235, 235, 235), "skin": (120, 150, 200), "navy": (90, 30, 25),
    "grey_dark": (60, 60, 60), "grey_light": (200, 200, 200),
}
BOX_A = [100, 100, 220, 400]
BOX_B = [400, 100, 520, 400]


def _frame(a_parts: dict, b_parts: dict, background=(90, 120, 90)) -> np.ndarray:
    """A canvas with two 'fighters' painted region by region."""
    image = np.zeros((520, 640, 3), dtype=np.uint8)
    image[:] = background
    for box, parts in ((BOX_A, a_parts), (BOX_B, b_parts)):
        x1, y1, x2, y2 = box
        height = y2 - y1
        bands = {"head": (0.0, 0.16), "torso": (0.16, 0.48), "shorts": (0.48, 0.70), "legs": (0.70, 1.0)}
        for name, (top, bottom) in bands.items():
            colour = parts.get(name, BGR["skin"])
            image[int(y1 + top * height):int(y1 + bottom * height), x1:x2] = colour
    return image


def _sim(a_parts, b_parts, background=(90, 120, 90)):
    return kit.kit_similarity(_frame(a_parts, b_parts, background), BOX_A, BOX_B)


def test_yellow_top_red_headgear_against_blue_top_blue_headgear_is_not_alike():
    result = _sim({"head": BGR["red"], "torso": BGR["yellow"], "shorts": BGR["red"]},
                  {"head": BGR["blue"], "torso": BGR["blue"], "shorts": BGR["blue"]})
    assert result["similarity"] < 0.2
    assert result["similarity"] < SETTINGS.max_kit_similarity


def test_black_top_against_blue_purple_top_is_not_alike():
    result = _sim({"torso": BGR["black"], "shorts": BGR["black"]},
                  {"torso": BGR["purple"], "shorts": BGR["blue"]})
    assert result["similarity"] < SETTINGS.max_kit_similarity


def test_identical_white_doboks_are_alike():
    dobok = {"head": BGR["skin"], "torso": BGR["white"], "shorts": BGR["white"], "legs": BGR["white"]}
    result = _sim(dobok, dict(dobok))
    assert result["similarity"] > 0.9
    assert result["similarity"] >= SETTINGS.max_kit_similarity
    assert kit.alike_reason(result) == "kit"


def test_matching_tops_but_different_headgear_can_be_told_apart():
    same_top = {"torso": BGR["white"], "shorts": BGR["white"]}
    result = _sim({**same_top, "head": BGR["red"]}, {**same_top, "head": BGR["blue"]})
    assert result["similarity"] < SETTINGS.max_kit_similarity
    assert result["most_different_region"] == "head"


def test_bare_chested_fighters_are_told_apart_by_their_shorts():
    result = _sim({"torso": BGR["skin"], "shorts": BGR["red"]}, {"torso": BGR["skin"], "shorts": BGR["blue"]})
    assert result["similarity"] < SETTINGS.max_kit_similarity
    assert result["most_different_region"] == "shorts"


def test_black_and_white_footage_is_named_as_the_cause():
    grey = {"head": BGR["grey_light"], "torso": BGR["grey_light"], "shorts": BGR["grey_dark"]}
    result = _sim(grey, dict(grey), background=(70, 70, 70))
    assert result["achromatic"]
    assert result["similarity"] >= SETTINGS.max_kit_similarity
    assert kit.alike_reason(result) == "black_and_white"


def test_black_and_white_with_different_trunks_is_still_separable():
    """Brightness is kept, so dark trunks against light trunks still separate."""
    result = _sim({"torso": BGR["grey_light"], "shorts": BGR["black"]},
                  {"torso": BGR["grey_light"], "shorts": BGR["white"]}, background=(70, 70, 70))
    assert result["similarity"] < SETTINGS.max_kit_similarity


def test_the_ring_and_crowd_do_not_make_different_kits_look_alike():
    """Only the centre line of each box is read, not the background."""
    busy = (40, 40, 200)
    result = _sim({"torso": BGR["yellow"], "shorts": BGR["yellow"]},
                  {"torso": BGR["blue"], "shorts": BGR["blue"]}, background=busy)
    assert result["similarity"] < 0.2


def test_tiny_fighters_are_named_as_the_cause():
    image = _frame({}, {})
    result = kit.kit_similarity(image, [100, 100, 110, 140], [400, 100, 410, 140])
    assert result["small"]
    assert kit.alike_reason(result) == "small"


@pytest.mark.parametrize("bad", [None])
def test_unreadable_input_is_none(bad):
    assert kit.kit_similarity(bad, BOX_A, BOX_B) is None


# --- identity no longer fails on kit alone; one cause, one recommendation ---

from core.report import identity_failure, identity_ready_by_fighter, times  # noqa: E402


def _tracking(**overrides):
    base = {
        "fighter_A_coverage": 0.9, "fighter_B_coverage": 0.9,
        "fighter_A_seed_source": "pose_detector", "fighter_B_seed_source": "pose_detector",
        "initial_iou_A": 0.9, "initial_iou_B": 0.9,
        "fighter_A_handoffs_per_minute": 1.0, "fighter_B_handoffs_per_minute": 1.0,
        "pair_similarity_method": "kit_regions_lab_v1",
    }
    base.update(overrides)
    return base


def test_identical_kit_that_tracking_held_is_trusted():
    tracking = _tracking(fighters_separable=False, fighter_pair_similarity=0.95,
                         kit_similarity={"similarity": 0.95}, identity_confusions=1,
                         identity_confusions_per_minute=0.5)
    assert identity_ready_by_fighter(tracking) == {"A": True, "B": True}
    assert identity_failure(tracking) is None


def test_identical_kit_that_tracking_could_not_hold_fails_with_that_cause():
    tracking = _tracking(fighters_separable=False, fighter_pair_similarity=0.95,
                         kit_similarity={"similarity": 0.95}, identity_confusions=14,
                         identity_confusions_per_minute=7.0)
    cause = identity_failure(tracking)
    assert cause["cause"] == "kit"
    assert "14 times" in cause["headline"] and "95% alike" in cause["headline"]
    assert cause["repick"] is True


def test_black_and_white_failure_does_not_blame_the_camera_or_offer_a_repick():
    tracking = _tracking(fighters_separable=False, fighter_pair_similarity=1.0,
                         kit_similarity={"similarity": 1.0, "achromatic": True},
                         identity_confusions=9, identity_confusions_per_minute=5.0)
    cause = identity_failure(tracking)
    assert cause["cause"] == "black_and_white"
    assert "no colour" in cause["headline"]
    assert "camera" not in cause["headline"].lower() and "busy hall" not in cause["headline"]
    assert cause["repick"] is False


def test_a_shaky_camera_is_one_recommendation_not_two():
    tracking = _tracking(fighter_A_suspicious_handoffs_per_minute=25.0, fighters_separable=False,
                         kit_similarity={"similarity": 0.9}, identity_confusions_per_minute=9.0)
    cause = identity_failure(tracking)
    assert cause["cause"] == "camera" and cause["repick"] is False
    assert "Show me who is who" not in cause["advice"]
    assert "Pick the two fighters again" not in cause["advice"]


def test_counts_read_as_english():
    assert times(1) == "once" and times(2) == "twice" and times(5) == "5 times"


def test_pair_check_uses_the_kit_measure_and_names_black_and_white(monkeypatch, tmp_path):
    import cv2

    import app.main as web
    from browser_client import BrowserClient

    grey = {"head": BGR["grey_light"], "torso": BGR["grey_light"], "shorts": BGR["grey_dark"]}
    frame = _frame(grey, dict(grey), background=(70, 70, 70))
    job_dir = web.OUTPUTS / "kitpaircheck1"
    job_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(job_dir / "selection.jpg"), frame)
    monkeypatch.setattr(web, "_authorized_job", lambda request, job_id: {"video_width": 640, "video_height": 520})
    with BrowserClient(web.app) as client:
        answer = client.post("/api/pair-check/kitpaircheck1",
                             json={"fighter_a_box": BOX_A, "fighter_b_box": BOX_B}).json()
    assert answer["looks_alike"] is True
    assert answer["cause"] == "black_and_white"
    assert "no colour" in answer["message"]
    assert "busy hall" not in answer["message"]

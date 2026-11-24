"""tools/strike_errors.py names why each strike was missed or counted falsely.

Run on the one fight whose pose track is in Git, against labels built from its
own counted strikes with known faults planted, so every answer is known.
"""

import json
import shutil
from pathlib import Path

import pytest

from tools import benchmark_dense_strikes as dense
from tools import strike_errors

TRACK = Path(__file__).resolve().parents[1] / "dataset" / "regression" / "kicklight_stavrou_ceschia" / "track.jsonl.gz"


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    fight = tmp_path_factory.mktemp("fight")
    shutil.copy(TRACK, fight / "track.jsonl.gz")
    counted = strike_errors.stages(fight / "track.jsonl.gz", None)["counted"]
    assert len(counted) >= 10
    labels = [{"t": float(e.peak_time), "fighter": e.fighter, "family": e.family, "sure": True} for e in counted[:6]]
    moved = counted[6]       # labelled as the other fighter's
    labels.append({"t": float(moved.peak_time), "fighter": "B" if moved.fighter == "A" else "A", "sure": True})
    end = float(counted[9].peak_time) + 0.1
    truth = {"covered": [0, end], "strikes": labels}
    (fight / "strikes.json").write_text(json.dumps(truth))
    (fight / "strikes.human.json").write_text(json.dumps(truth))
    return fight, moved, counted


def test_totals_are_the_benchmarks_own(planted):
    fight, _, _ = planted
    mine = strike_errors.diagnose(fight)
    theirs = dense.score(dense.counted_events(dense.replay(fight / "track.jsonl.gz"), None), dense.load_truth(fight))
    assert (mine["labelled"], mine["counted"], mine["real"]) == (theirs["labelled"], theirs["counted"], theirs["real"])
    assert len(mine["misses"]) == theirs["missed"]
    assert len(mine["false"]) == theirs["false"]


def test_a_strike_given_to_the_wrong_fighter_is_named_both_ways(planted):
    fight, moved, _ = planted
    result = strike_errors.diagnose(fight)
    t = round(float(moved.peak_time), 2)
    assert any(m["t"] == t and m["why"] == "proposed for the other fighter" for m in result["misses"])
    assert any(f["t"] == t and f["why"] == "the other fighter's strike" for f in result["false"])


def test_a_count_with_nothing_labelled_near_it_says_so(planted):
    fight, _, counted = planted
    result = strike_errors.diagnose(fight)
    unlabelled = {round(float(e.peak_time), 2) for e in counted[7:10]} - {round(float(counted[6].peak_time), 2)}
    named = {f["t"]: f["why"] for f in result["false"]}
    assert any(named.get(t) in {"nothing labelled there", "the other fighter's strike", "a second count of one strike"}
               for t in unlabelled)


def test_the_output_carries_numbers_only():
    """Pasted into a chat: no file names of videos, no pictures."""
    source = Path(strike_errors.__file__).read_text(encoding="utf-8")
    assert "keypoints" not in source.split("def _brief")[1]

"""The free Ringside Scorer (owner, 2026-11-02): core/ringside.py and /scorer."""

from __future__ import annotations

import re

import pytest

from core import ringside
from core.scoring import RULESETS


def _keys(ruleset):
    return [button["key"] for button in ringside.buttons(ruleset)]


def test_every_ruleset_offers_only_what_its_scorer_counts():
    assert _keys("BOXING") == ["punch_head", "punch_body"]                 # no kicks in boxing
    assert "punch_head" not in _keys("WT_TAEKWONDO")                       # head punches illegal in WT
    assert "knee_body" not in _keys("LOW_KICK")                            # no knees in Low Kick
    assert "kick_leg" not in _keys("ITF_TAEKWONDO")                        # a leg scores nothing in ITF
    assert "elbow" in _keys("MUAY_THAI") and "elbow" not in _keys("MUAY_THAI_NO_ELBOWS")
    for key in RULESETS:
        assert ringside.buttons(key), key
        assert ringside.rules_summary(key)


def test_points_rulesets_use_the_federation_table():
    values = {b["key"]: b["points"] for b in ringside.buttons("KICK_LIGHT")}
    assert values["punch_head"] == 1 and values["kick_head"] == 2 and values["jump_kick_head"] == 3
    card = ringside.score("KICK_LIGHT", [{"A": ["punch_head", "kick_head"], "B": ["kick_body"]}])
    assert card["totals"] == {"A": 3, "B": 1} and card["winner"] == "A"
    wt = {b["key"]: b["points"] for b in ringside.buttons("WT_TAEKWONDO")}
    assert wt["turn_kick_head"] == 6


def test_ten_point_must_rounds_and_knockdowns():
    card = ringside.score("BOXING", [{"A": ["punch_head"] * 6, "B": ["punch_body"] * 3},
                                     {"A": ["punch_head"], "B": ["punch_head"] * 2, "knockdowns": {"A": 1}}])
    assert [(r["A"], r["B"]) for r in card["rounds"]] == [(10, 9), (10, 8)]   # a knockdown wins the round
    assert card["totals"] == {"A": 20, "B": 17}
    muay_thai = ringside.score("MUAY_THAI", [{"A": ["kick_body"] * 10, "B": ["punch_body"]}])
    assert (muay_thai["rounds"][0]["A"], muay_thai["rounds"][0]["B"]) == (10, 8)  # IFMA: a lead of 9 is 10-8
    even = ringside.score("MMA", [{"A": ["punch_body"], "B": ["punch_body"]}])
    assert even["rounds"][0]["winner"] == "EVEN"


def test_unknown_taps_are_ignored_and_bad_rulesets_refused():
    card = ringside.score("BOXING", [{"A": ["kick_head", "punch_head"], "B": []}])
    assert card["rounds"][0]["landed_A"] == 1
    with pytest.raises(ValueError):
        ringside.score("NOT_A_SPORT", [])


def test_the_page_and_the_score_api():
    from fastapi.testclient import TestClient

    import app.main as web

    client = TestClient(web.app)
    page = client.get("/scorer")
    assert page.status_code == 200 and "Ringside Scorer" in page.text and "/static/ringside.js" in page.text
    assert "/scorer" in web.PUBLIC_INDEX_ROUTES
    refused = client.post("/api/ringside/score", json={"ruleset": "BOXING", "rounds": []})
    assert refused.status_code in {400, 403}                            # needs the CSRF token
    meta = re.search(r'name="csrf-token" content="([^"]+)"', page.text)
    scored = client.post("/api/ringside/score", json={"ruleset": "K1", "rounds": [{"A": ["kick_leg"], "B": []}]},
                         headers={"X-CSRF-Token": meta.group(1)})
    assert scored.status_code == 200 and scored.json()["totals"] == {"A": 1, "B": 0}

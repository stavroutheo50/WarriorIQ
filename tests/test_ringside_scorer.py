"""The free Ringside Scorer: core/ringside.py and /scorer.

+1 / +2 per corner and a point deduction (owner, 2026-11-09)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core import ringside
from core.scoring import RULESETS


def _round(a=0, b=0, minus_a=0, minus_b=0):
    return {"A": {"points": a, "deductions": minus_a}, "B": {"points": b, "deductions": minus_b}}


def test_every_ruleset_has_a_summary():
    for key in RULESETS:
        assert ringside.rules_summary(key)


def test_points_rules_add_up_taps_less_deductions():
    card = ringside.score("KICK_LIGHT", [_round(5, 3, minus_a=1), _round(2, 4)])
    assert [(r["A"], r["B"]) for r in card["rounds"]] == [(4, 3), (2, 4)]
    assert card["totals"] == {"A": 6, "B": 7} and card["winner"] == "B"


def test_ten_point_must_rounds_and_deductions():
    card = ringside.score("BOXING", [_round(6, 3), _round(2, 2), _round(1, 4, minus_b=1)])
    assert [(r["A"], r["B"]) for r in card["rounds"]] == [(10, 9), (10, 10), (9, 9)]
    assert [r["winner"] for r in card["rounds"]] == ["A", "EVEN", "EVEN"]
    assert card["totals"] == {"A": 29, "B": 28}
    assert ringside.score("MMA", [_round(1, 1)])["rounds"][0]["winner"] == "EVEN"


def test_bad_input_is_ignored_and_bad_rulesets_refused():
    card = ringside.score("BOXING", [{"A": {"points": "x", "deductions": -3}, "B": ["old", "taps"]}])
    assert card["rounds"][0]["points_A"] == 0 and card["rounds"][0]["deductions_A"] == 0
    assert card["rounds"][0]["points_B"] == 0
    assert len(ringside.score("K1", [_round()] * 40)["rounds"]) == ringside.MAX_ROUNDS
    with pytest.raises(ValueError):
        ringside.score("NOT_A_SPORT", [])


def test_the_page_only_offers_points_and_a_deduction():
    script = (Path(__file__).resolve().parents[1] / "app" / "static" / "ringside.js").read_text(encoding="utf-8")
    assert "[1, 2].forEach" in script and "Point deduction" in script
    assert "knockdown" not in script.lower() and "button.label" not in script


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
    scored = client.post("/api/ringside/score", json={"ruleset": "KICK_LIGHT", "rounds": [_round(2, 0)]},
                         headers={"X-CSRF-Token": meta.group(1)})
    assert scored.status_code == 200 and scored.json()["totals"] == {"A": 2, "B": 0}

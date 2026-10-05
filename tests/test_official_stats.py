"""WarriorIQ's strike counts compared with a bout's official statistics."""

import pytest

from core.official_stats import compare, official_rounds, warrioriq_rounds


def _row(fighter, round_number, total="20 of 40", sig="15 of 30", head="8 of 20", body="4 of 6",
         leg="3 of 4", kd="0", td="1 of 2", event="UFC 1", bout="Ana Lima vs. Bea Ruiz"):
    return {"EVENT": event, "BOUT": bout, "ROUND": f"Round {round_number}", "FIGHTER": fighter,
            "KD": kd, "SIG.STR.": sig, "TOTAL STR.": total, "TD": td,
            "HEAD": head, "BODY": body, "LEG": leg}


ROWS = [_row("Ana Lima", 1), _row("Ana Lima", 2, total="10 of 30"), _row("Béa Ruiz", 1, total="5 of 9")]


def _strike(fighter, round_number, outcome="landed", target="head", family="punch"):
    return {"fighter": fighter, "round_number": round_number, "outcome": outcome,
            "target": target, "family": family}


def test_official_lines_are_read_per_round_ignoring_case_and_accents():
    lines = official_rounds(ROWS, bout="ana lima vs bea ruiz", fighter="Bea Ruiz")
    assert list(lines) == [1]
    assert (lines[1].total.landed, lines[1].total.attempted) == (5, 9)
    lines = official_rounds(ROWS, bout="Ana Lima vs. Bea Ruiz", fighter="Ana Lima")
    assert (lines[2].total.landed, lines[2].total.attempted) == (10, 30)
    assert (lines[1].by_target["leg"].landed, lines[1].takedowns.attempted) == (3, 2)


def test_a_rematch_needs_the_event_named():
    rows = ROWS + [_row("Ana Lima", 1, event="UFC 2")]
    with pytest.raises(LookupError, match="more than once"):
        official_rounds(rows, bout="Ana Lima vs. Bea Ruiz", fighter="Ana Lima")
    assert official_rounds(rows, bout="Ana Lima vs. Bea Ruiz", fighter="Ana Lima", event="UFC 2")


def test_an_unknown_bout_is_an_error_not_zero_strikes():
    with pytest.raises(LookupError):
        official_rounds(ROWS, bout="Nobody vs. Nobody", fighter="Ana Lima")


def test_only_strikes_by_that_fighter_are_counted_and_outcomes_normalised():
    events = [_strike("A", 1), _strike("A", 1, outcome="clean", target="leg"),
              _strike("A", 1, outcome="blocked"), _strike("A", 1, outcome="missed", family="kick"),
              _strike("B", 1), {"fighter": "A", "round_number": 1, "family": "down"}]
    line = warrioriq_rounds(events, "A")[1]
    assert (line.total.landed, line.total.attempted) == (2, 4)
    assert (line.by_target["head"].landed, line.by_target["leg"].landed) == (1, 1)


def test_comparison_reports_both_official_definitions_and_missing_rounds():
    official = official_rounds(ROWS, bout="Ana Lima vs. Bea Ruiz", fighter="Ana Lima")
    ours = warrioriq_rounds([_strike("A", 1)] * 18 + [_strike("A", 1, outcome="missed")] * 20, "A")
    result = compare(ours, official)
    first = result["rounds"][0]
    assert first["landed_vs_total"] == {"warrioriq": 18, "official": 20, "difference": -2, "relative": -0.1}
    assert first["landed_vs_significant"]["official"] == 15
    assert first["attempted_vs_total"]["difference"] == -2
    # Round 2 was not analysed: listed as missing, never counted as zero strikes.
    assert result["rounds_missing_from_warrioriq"] == [2]
    assert result["totals"]["landed_vs_total"]["official"] == 20

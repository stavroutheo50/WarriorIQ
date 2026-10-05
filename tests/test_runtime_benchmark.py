import pytest

from tools.benchmark_analysis import StageTimers, timing_summary


def run(seconds, fingerprint="same"):
    return {"wall_seconds": seconds, "output_hashes": {"events.json": fingerprint, "tracking.jsonl": fingerprint}}


def test_first_run_is_not_in_warm_median():
    summary = timing_summary([run(100), run(12), run(10), run(11)])
    assert summary["first_run_seconds"] == 100
    assert summary["warm_median_seconds"] == 11
    assert summary["warm_min_seconds"] == 10
    assert summary["warm_max_seconds"] == 12
    assert summary["warm_runs"] == 3
    assert summary["outputs_identical_across_repeats"]


def test_changed_predictions_are_not_claimed_identical():
    assert not timing_summary([run(10), run(8, "changed")])["outputs_identical_across_repeats"]


def test_single_run_cannot_establish_warm_performance():
    summary = timing_summary([run(10)])
    assert summary["warm_median_seconds"] is None
    assert summary["warm_runs"] == 0


@pytest.mark.parametrize("values", [[], [run(0)], [run(-1)], [run(float("nan"))], [run(float("inf"))]])
def test_invalid_measurements_are_rejected(values):
    with pytest.raises(ValueError):
        timing_summary(values)


def test_nested_stage_time_is_not_double_counted():
    clock = iter([0, 1, 3, 5])
    timer = StageTimers(clock=lambda: next(clock))
    inner = timer.wrap(lambda: "original result", "inner")
    outer = timer.wrap(inner, "outer")
    assert outer() == "original result"
    assert timer.rows["inner"]["exclusive_seconds"] == 2
    assert timer.rows["outer"]["exclusive_seconds"] == 3


def test_stage_timer_preserves_exception_and_cleans_stack():
    clock = iter([0, 2])
    timer = StageTimers(clock=lambda: next(clock))

    def fail():
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        timer.wrap(fail, "failed")()
    assert timer.stack == []
    assert timer.rows["failed"]["exclusive_seconds"] == 2

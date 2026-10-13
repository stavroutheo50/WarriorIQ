"""The slow-run watchdog (core/slow_watch.py).

2026-10-13: a fight ran five times slower than its own start on an RTX 5060
and the report could not say why.
"""

from core.slow_watch import SLOW_FACTOR, SlowRunWatch, slow_reasons


def _run(watch, per_frame, frames, start=0.0):
    t = start
    for i in range(frames):
        t += per_frame
        watch.frame(t, i * 0.2)
    return t


def test_a_steady_run_reads_nothing():
    calls = []
    watch = SlowRunWatch(snapshot=lambda: calls.append(1) or {})
    _run(watch, 0.05, 200)
    assert watch.summary() is None and calls == []


def test_a_run_that_slows_down_reads_the_machine_once_and_says_why():
    calls = []
    snapshot = {"gpu": {"temperature_c": 88.0, "throttle_reasons": "0x20"},
                "other_gpu_programs": [{"name": "game.exe", "memory_mb": 3000.0}],
                "memory": {"available_bytes": 800 * 1024 ** 2}, "cpu_percent": 40.0}
    watch = SlowRunWatch(snapshot=lambda: calls.append(1) or snapshot)
    t = _run(watch, 0.05, 60)
    _run(watch, 0.05 * (SLOW_FACTOR + 1), 60, start=t)
    event = watch.summary()
    assert calls == [1]
    assert event["times_slower"] >= SLOW_FACTOR
    assert any("hot (88" in r for r in event["reasons"])
    assert any("game.exe" in r for r in event["reasons"])
    assert any("low on memory" in r for r in event["reasons"])


def test_nothing_readable_means_no_invented_reason():
    assert slow_reasons({}) == []
    assert slow_reasons({"gpu": {"temperature_c": 60.0, "throttle_reasons": "0x0000000000000000"},
                         "other_gpu_programs": [], "cpu_percent": 20.0}) == []

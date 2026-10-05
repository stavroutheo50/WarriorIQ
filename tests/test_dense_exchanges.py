"""Optional denser sampling while the fighters are within striking range."""

from types import SimpleNamespace
from unittest import mock

from core import analyzer


def _fighter(x):
    return SimpleNamespace(box=[x, 0, x + 50, 200])


def test_close_fighters_are_in_range_and_distant_ones_are_not():
    assert analyzer._fighters_in_range(_fighter(0), _fighter(150))       # 0.75 body heights apart
    assert not analyzer._fighters_in_range(_fighter(0), _fighter(600))   # 3 body heights apart
    assert not analyzer._fighters_in_range(_fighter(0), None)


def test_off_by_default_the_stride_never_changes():
    assert analyzer._exchange_stride(4, True) == 4


def test_on_the_stride_halves_only_during_an_exchange():
    on = SimpleNamespace(dense_exchange_sampling=True, dense_exchange_body_lengths=1.5)
    with mock.patch.object(analyzer, "SETTINGS", on):
        assert analyzer._exchange_stride(4, True) == 2
        assert analyzer._exchange_stride(4, False) == 4
        assert analyzer._exchange_stride(1, True) == 1

"""The cloud GPU backs the analysis PC up; it does not run every fight."""

from types import SimpleNamespace
from unittest import mock

from app import main


def _settings(delay):
    return mock.patch.object(main, "SETTINGS", SimpleNamespace(cloud_fallback_after_seconds=delay))


def test_without_a_delay_the_cloud_is_woken_at_once_as_before():
    slept = []
    with _settings(0):
        assert main._cloud_still_needed("job1", sleep=slept.append)
    assert slept == []


def test_a_fight_the_pc_claimed_does_not_start_the_cloud():
    slept = []
    with _settings(120), mock.patch.object(main, "get_job", return_value={"status": "running"}):
        assert not main._cloud_still_needed("job1", sleep=slept.append)
    assert slept == [120]


def test_a_fight_still_waiting_starts_the_cloud():
    with _settings(120), mock.patch.object(main, "get_job", return_value={"status": "queued"}):
        assert main._cloud_still_needed("job1", sleep=lambda _: None)


def test_a_fight_that_vanished_does_not_start_the_cloud():
    with _settings(120), mock.patch.object(main, "get_job", return_value=None):
        assert not main._cloud_still_needed("job1", sleep=lambda _: None)

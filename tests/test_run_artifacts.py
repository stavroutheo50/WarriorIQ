import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest

from app import state


def _late_worker_update(output_root, run_id, ready, finished, result):
    state.OUTPUTS = Path(output_root)
    ready.set()
    result.put(state.update_job_for_worker("fight", "gpu", run_id, {"percent": 99}))
    finished.set()


@pytest.fixture
def output_root(tmp_path):
    with patch.object(state, "OUTPUTS", tmp_path), patch.object(state, "_jobs", {}):
        yield tmp_path


def _running_job():
    state.create_job("fight", {"video_path": "source.mp4"})
    run_id = state.prepare_job_run("fight", {})
    assert state.start_job_run("fight", "gpu", run_id)
    return run_id


def test_new_run_hides_but_preserves_previous_legacy_artifacts(output_root):
    state.create_job("fight", {"status": "complete", "analysis_run_id": "a" * 32})
    directory = output_root / "fight"
    for name in ("report.json", "tracking.jsonl", "events.json"):
        (directory / name).write_text("old data", encoding="utf-8")
    assert state.completed_artifact_directory("fight") == directory

    state.prepare_job_run("fight", {})

    assert state.completed_artifact_directory("fight") is None
    assert (directory / "report.json").read_text(encoding="utf-8") == "old data"
    assert (directory / "tracking.jsonl").read_text(encoding="utf-8") == "old data"


def test_committed_runs_use_distinct_artifacts_even_if_old_worker_keeps_writing(output_root):
    first_run = _running_job()
    first_directory = state.analysis_run_directory("fight", first_run)
    first_directory.mkdir(parents=True)
    first_tracking = first_directory / "tracking.jsonl"
    first_tracking.write_text("first skeleton", encoding="utf-8")
    assert state.finalize_job_from_worker(
        "fight", "gpu", first_run, {"run": "first"}, {"tracking.jsonl": first_tracking},
    )
    assert state.completed_artifact_directory("fight") == first_directory

    second_run = state.prepare_job_run("fight", {})
    assert state.completed_artifact_directory("fight") is None
    assert state.start_job_run("fight", "gpu", second_run)
    assert state.finalize_job_from_worker("fight", "gpu", second_run, {"run": "second"}, {})
    first_tracking.write_text("late first skeleton", encoding="utf-8")
    assert not state.finalize_job_from_worker("fight", "gpu", first_run, {"run": "stale"}, {})

    current = state.completed_artifact_directory("fight")
    assert current != first_directory
    assert json.loads((current / "report.json").read_text(encoding="utf-8")) == {
        "run": "second", "analysis_run_id": second_run,
    }
    assert not (current / "tracking.jsonl").exists()


def test_failed_artifact_publication_never_exposes_partial_result(output_root):
    run_id = _running_job()
    source = output_root / "incoming-tracking"
    source.write_text("skeleton", encoding="utf-8")

    assert not state.finalize_job_from_worker("fight", "gpu", run_id, {"run": "new"}, {
        "tracking.jsonl": source,
        "events.json": output_root / "missing-events",
    })

    assert state.get_job("fight")["status"] == "running"
    assert state.completed_artifact_directory("fight") is None
    assert (state.analysis_run_directory("fight", run_id) / "tracking.jsonl").exists()


def test_failed_session_commit_does_not_expose_completed_artifacts(output_root):
    run_id = _running_job()
    with patch.object(state, "_write_session", return_value=False):
        assert not state.finalize_job_from_worker("fight", "gpu", run_id, {"run": "new"}, {})
    assert state.get_job("fight")["status"] == "running"
    assert state.completed_artifact_directory("fight") is None
    assert state._jobs["fight"]["status"] == "running"


def test_uncommitted_complete_patch_cannot_publish_a_run(output_root):
    run_id = _running_job()
    assert state.update_job_for_worker("fight", "gpu", run_id, {"status": "complete"})
    assert state.completed_artifact_directory("fight") is None


def test_legacy_saved_report_without_runtime_session_stays_readable(output_root):
    directory = output_root / "historic"
    directory.mkdir()
    (directory / "report.json").write_text("{}", encoding="utf-8")
    assert state.completed_artifact_directory("historic") == directory


def test_corrupt_session_cannot_resurrect_a_legacy_report(output_root):
    state.create_job("fight", {"status": "complete"})
    (output_root / "fight" / "report.json").write_text('{"old":true}')
    (output_root / "fight" / "analysis-session.json").write_text("broken")
    assert state.completed_artifact_directory("fight") is None


def test_inprocess_queue_also_requires_durable_state(output_root):
    state.create_job("fight", {"status": "complete"})
    with patch.object(state, "_write_session", return_value=False):
        with pytest.raises(state.AnalysisStateNotPersisted):
            state.prepare_job_run("fight", {})


def test_artifact_names_cannot_escape_run_directory(output_root):
    run_id = _running_job()
    with pytest.raises(ValueError, match="Unsupported analysis artifact"):
        state.finalize_job_from_worker("fight", "gpu", run_id, {}, {"../report.json": output_root / "x"})
    with pytest.raises(ValueError, match="Invalid analysis generation"):
        state.analysis_run_directory("fight", "../previous")


def test_history_failure_preserves_completed_report_and_can_retry(output_root):
    run_id = _running_job()
    state.update_job("fight", {"persist_result": True})
    assert state.finalize_job_from_worker("fight", "gpu", run_id, {"events": []}, {})
    with patch("core.db.save_completed_analysis", side_effect=OSError("database unavailable")):
        assert not state.persist_completed_job("fight", run_id)
    assert state.get_job("fight")["status"] == "complete"
    assert state.get_job("fight")["history_saved"] is False
    assert state.completed_artifact_directory("fight") is not None
    with patch("core.db.save_completed_analysis") as save:
        assert state.persist_completed_job("fight", run_id)
        assert state.persist_completed_job("fight", run_id)
        save.assert_called_once()
        assert save.call_args.args[2]["analysis_run_id"] == run_id


def test_previous_generation_cannot_overwrite_history(output_root):
    first = _running_job()
    state.update_job("fight", {"persist_result": True})
    assert state.finalize_job_from_worker("fight", "gpu", first, {}, {})
    second = state.prepare_job_run("fight", {})
    assert state.start_job_run("fight", "gpu", second)
    assert state.finalize_job_from_worker("fight", "gpu", second, {}, {})
    with patch("core.db.save_completed_analysis") as save:
        assert not state.persist_completed_job("fight", first)
        save.assert_not_called()


def test_separate_process_cannot_overwrite_newer_generation(output_root):
    run_id = _running_job()
    context = multiprocessing.get_context("spawn")
    ready, finished = context.Event(), context.Event()
    result = context.Queue()
    process = context.Process(
        target=_late_worker_update,
        args=(str(output_root), run_id, ready, finished, result),
    )
    try:
        with state._job_lock("fight"):
            process.start()
            assert ready.wait(10), "worker failed to initialize"
            assert not finished.wait(0.2), "worker bypassed the process lock"
            job = state._read_session("fight")
            job.update({"analysis_run_id": "b" * 32, "percent": 0})
            assert state._write_session("fight", job)
        process.join(10)
        assert process.exitcode == 0
        assert result.get(timeout=2) is False
        assert state.get_job("fight")["percent"] == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        result.close()

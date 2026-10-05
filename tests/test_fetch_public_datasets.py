"""Every public dataset source states its licence and whether it may be used commercially."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_public_datasets", ROOT / "tools" / "fetch_public_datasets.py")
fetch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fetch                     # dataclasses look their module up
spec.loader.exec_module(fetch)


def test_every_source_records_its_licence_and_commercial_status():
    names = [source.name for source in fetch.SOURCES]
    assert len(names) == len(set(names))
    for source in fetch.SOURCES:
        assert source.licence and source.homepage.startswith("https://")
        assert source.commercial_use in {"yes", "no", "unclear"}


def test_fetched_data_is_never_committed():
    assert "dataset/public/" in (ROOT / ".gitignore").read_text()


def test_a_source_missing_its_credentials_is_skipped_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    kaggle = next(s for s in fetch.SOURCES if s.kind == "kaggle")
    assert "KAGGLE" in fetch._fetch(kaggle, tmp_path / kaggle.name)

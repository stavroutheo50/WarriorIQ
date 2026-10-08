"""Render a template the way the app does, for tests that check page text."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "app" / "templates"


class _Stub:
    def __init__(self, **kw): self.__dict__.update(kw)
    def __getattr__(self, key): return _Stub()
    def __getitem__(self, key): return _Stub()
    def __str__(self): return ""
    def __bool__(self): return False
    def __iter__(self): return iter(())


def _env():
    from jinja2 import ChainableUndefined, Environment, FileSystemLoader

    from app.main import templates as app_templates

    env = Environment(loader=FileSystemLoader(str(ROOT)), undefined=ChainableUndefined)
    env.filters.update(app_templates.env.filters)
    env.globals.update(app_templates.env.globals)
    return env


def request_stub(account=None):
    return _Stub(url=_Stub(path="/"), state=_Stub(account=account), cookies={}, headers={})


def render(name: str, account=None, **context) -> str:
    return _env().get_template(name).render(request=request_stub(account), **context)


def render_result(**context) -> str:
    from app.main import _analysis_quality_summary, sport_identity

    report = json.loads((Path(__file__).resolve().parent / "fixtures" / "report_sample.json")
                        .read_text(encoding="utf-8"))
    base = {"job_id": "abc", "report": report, "identity": sport_identity("kickboxing"),
            "report_access": {"report_tier": "full", "report_label": "Full", "label": "Full"},
            "analysis_quality": _analysis_quality_summary(report), "unavailable": []}
    base.update(context)
    return render("result.html", **base)

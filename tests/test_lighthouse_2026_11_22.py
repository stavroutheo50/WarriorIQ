"""Accessibility findings from a Lighthouse pass over the signed-in pages (2026-11-22).

Each was a failing axe rule: a form field with no label, a heading level
skipped, a table cell with no header, a <dl> holding a <div> that is not a
dt/dd group, and white text on the light red "Avoid" badge (2.99:1).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "app" / "templates"


def _t(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def test_profile_fields_have_labels_and_headings_do_not_skip_a_level():
    page = _t("profile.html")
    assert '<label for="profileDisplayName">' in page and 'id="profileDisplayName" name="display_name"' in page
    assert '<label for="profileNotes">' in page and 'id="profileNotes" name="notes"' in page
    assert '<div class="profile-head">' in page and '<h3 style="margin:0">' not in page


def test_stat_bars_are_dd_cells_inside_the_definition_list():
    sheet = _t("_stat_sheet.html")
    assert '<dd class="stat-bar" aria-hidden="true">' in sheet
    assert '<div class="stat-bar"' not in sheet


def test_every_squad_table_column_has_a_header():
    camp = _t("camp.html")
    assert "<th></th>" not in camp
    assert camp.count('<th><span class="visually-hidden">Report</span></th>') == 2


def test_the_avoid_badge_text_is_dark_on_the_red():
    glass = (ROOT / "app" / "static" / "glass.css").read_text(encoding="utf-8")
    assert ".film-shot.is-bad figcaption b { background: #ff5d6c; color: #06121a; }" in glass


def test_hashed_css_and_js_are_cached_for_a_year_and_everything_else_for_a_week():
    """The ?v= token hashes the top-level CSS and JS, so only those, at the current token, are immutable."""
    import app.main as web
    from fastapi.testclient import TestClient

    client = TestClient(web.app)
    v = web.ASSET_VERSION
    year, week = "public, max-age=31536000, immutable", "public, max-age=604800"
    assert client.get(f"/assets/base.css?v={v}").headers["cache-control"] == year
    assert client.get(f"/static/home.js?v={v}").headers["cache-control"] == year
    assert client.get(f"/static/components.css?v={v}").headers["cache-control"] == year
    # A stale or missing token, an image or a font: a week, as before.
    assert client.get("/assets/base.css?v=old").headers["cache-control"] == week
    assert client.get("/static/home.js").headers["cache-control"] == week
    assert client.get(f"/static/hero-fight.jpg?v={v}").headers["cache-control"] == week
    assert client.get("/static/fonts/inter-var.woff2").headers["cache-control"] == week

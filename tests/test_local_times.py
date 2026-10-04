"""Every time on the site is shown in the viewer's own timezone (QA, 2026-10-04).

Compare showed UTC while Profile showed local time for the same fight: the
compare picker's labels are <option> text, which cannot hold a <time>. Those
carry the UTC part in data-local-stamp and base.html swaps just that part.
"""

from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "templates"


def test_the_compare_picker_and_headings_are_localised():
    page = (TEMPLATES / "compare.html").read_text(encoding="utf-8")
    assert 'data-local-at="{{f.created_at}}" data-local-stamp="{{f.choice_stamp}}"' in page
    assert 'data-local-at="{{pick.created_at}}" data-local-stamp="{{pick.choice_stamp}}"' in page
    base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert "[data-local-at][data-local-stamp]" in base


def test_the_stamp_is_exactly_the_text_inside_the_label():
    from core.squad import fight_choice_label, fight_choice_stamp

    created = "2026-10-04T21:30:00+00:00"
    stamp = fight_choice_stamp(created)
    assert stamp and stamp in fight_choice_label("K1", created, "competition", "Theo")


def test_no_raw_timestamp_is_left_in_utc():
    for name, needle in (("settings.html", "Requested <time datetime=\"{{action.requested_at}}\" data-local>"),
                         ("shared.html", "Link expires <time datetime=\"{{ expires_at }}\" data-local=\"date\">"),
                         ("result.html", "data-local=\"date\">{{sharing.links[0].expires_label}}</time>")):
        assert needle in (TEMPLATES / name).read_text(encoding="utf-8"), name

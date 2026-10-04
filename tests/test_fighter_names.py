"""Reports use the filed fighter's name (QA, 2026-10-04).

The library showed "Theodoulos" while the report said "Fighter A / Fighter B".
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import patch


def test_the_filed_fighter_names_their_side_and_the_other_is_the_opponent():
    import app.main as web

    request = SimpleNamespace()
    with patch.object(web, "_profile_id", return_value=7), \
            patch.object(web, "get_fighter", return_value={"name": "  Theodoulos   Stavrou "}) as get:
        names = web._fighter_names(request, {"fighter_id": "12", "focus_fighter": "B"}, {})
    get.assert_called_once_with(7, 12)
    assert names == {"A": "Opponent", "B": "Theodoulos Stavrou"}


def test_without_a_filed_fighter_the_letters_stay():
    import app.main as web

    with patch.object(web, "_profile_id", return_value=7):
        assert web._fighter_names(SimpleNamespace(), {}, {}) == {"A": "Fighter A", "B": "Fighter B"}
    with patch.object(web, "_profile_id", return_value=None):
        assert web._fighter_names(SimpleNamespace(), {"fighter_id": "3"}, {})["A"] == "Fighter A"


def test_the_report_page_prints_the_name_not_the_letter():
    from test_web import NoPunchClaimLeaksTests

    import app.main as web

    report = NoPunchClaimLeaksTests._report(trusted=True)
    names = {"A": "Theodoulos", "B": "Opponent"}
    web._name_the_fighters(report, names)
    html = NoPunchClaimLeaksTests._render(report, names=names)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    assert "Theodoulos" in text
    assert "Performance report · Theodoulos" in text
    # Only the technical diagnostics ("Performance integrity") keep the
    # internal letters.
    visible = text.split("Performance integrity")[0]
    assert "Fighter A" not in visible

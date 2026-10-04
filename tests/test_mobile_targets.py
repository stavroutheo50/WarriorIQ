"""Phone tap targets and overflow (QA, 2026-10-04, item 21).

tools/audit_mobile_layout.py was run against every page at 390 px, signed in
and out (35 pages): before, every page had targets under 44 px (32 px brand,
42 px menu button, 23 px FAQ questions, 15 px pricing links, 31 px fields)
and none overflowed; after, no page reports a problem. These pin the rules.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "app" / "static" / "product.css").read_text(encoding="utf-8")


def test_the_phone_rules_cover_every_kind_of_target():
    block = CSS.split("Phone tap targets: 44px minimum", 1)[1]
    for selector in (".nav .brand{min-height:44px", ".mobile-menu-button{min-width:44px;min-height:44px}",
                     "summary{min-height:44px", "label:has(> input[type=radio])", "aside a,nav a",
                     "th a,.record-delete{min-width:44px"):
        assert selector in block, selector


def test_the_audit_tool_is_kept():
    tool = (ROOT / "tools" / "audit_mobile_layout.py").read_text(encoding="utf-8")
    assert "r.height < 43.5 || r.width < 43.5" in tool and "fs < 12" in tool and "docW" in tool

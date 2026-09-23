"""The stylesheet squeeze, and the things it must refuse to touch.

216 KB of CSS reached every page on a site whose heaviest page carries one
image. The saving is worth having; a minifier that is clever about CSS it does
not parse is not, because it breaks one page on one browser and nobody finds
out until a phone loads it.

So the tests here are mostly about what is left alone.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from core.config import SETTINGS
from core.css_minify import minify

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LeavesMeaningAloneTests(unittest.TestCase):
    def test_a_string_is_not_a_declaration(self):
        """It was rewriting text. `content:"a, b"` came back as `"a,b"`, and
        `content:";}"` lost its semicolon, because a final tidy-up pass ran
        across the protected tokens after they had been put back."""
        self.assertEqual(minify('.a{content:"a, b"}'), '.a{content:"a, b"}')
        self.assertEqual(minify('.b{content:";}"}'), '.b{content:";}"}')
        self.assertEqual(minify(".c{content:'x, y'}"), ".c{content:'x, y'}")

    def test_calc_keeps_the_spaces_it_needs(self):
        """calc(100% - 20px) without its spaces is not calc at all, and it is
        the classic way to break a layout silently."""
        for source in (".x{width:calc(100% - 20px)}",
                       ".y{width:calc(100% + 2px)}",
                       ".z{margin:calc(1rem - 2px) auto}"):
            with self.subTest(source=source):
                self.assertEqual(minify(source), source)

    def test_a_descendant_selector_is_not_a_pseudo_selector(self):
        """`li :first-child` and `li:first-child` select different elements."""
        self.assertEqual(minify("li :first-child{color:red}"), "li :first-child{color:red}")
        self.assertEqual(minify("li:first-child{color:red}"), "li:first-child{color:red}")

    def test_a_data_uri_is_opaque(self):
        source = ".g{background:url(data:image/png;base64,A{B,C)}"
        self.assertEqual(minify(source), source)

    def test_value_lists_keep_their_spaces(self):
        self.assertEqual(minify(".i{margin:0 auto;padding:1px 2px}"),
                         ".i{margin:0 auto;padding:1px 2px}")

    def test_important_survives(self):
        self.assertIn("!important", minify(".f{color:red!important}"))

    def test_a_media_query_still_parses(self):
        self.assertEqual(minify("@media (max-width:620px){.d{color:red}}"),
                         "@media (max-width:620px){.d{color:red}}")


class DoesTheWorkTests(unittest.TestCase):
    def test_comments_go(self):
        self.assertNotIn("gone", minify("/* gone */.c{color:red}"))

    def test_a_comment_between_two_tokens_leaves_a_space(self):
        """`a/* x */b` is a descendant selector. Removing the comment outright
        would glue it into one element name."""
        self.assertEqual(minify("a/* x */b{color:red}"), "a b{color:red}")

    def test_whitespace_around_delimiters_goes(self):
        self.assertEqual(minify("a { color:red ; }"), "a{color:red}")

    def test_the_last_semicolon_goes(self):
        self.assertEqual(minify(".a{color:red;}"), ".a{color:red}")

    def test_it_actually_shrinks_the_real_bundles(self):
        static = PROJECT_ROOT / "app" / "static"
        raw = "\n".join(p.read_text(encoding="utf-8") for p in sorted(static.glob("*.css")))
        squeezed = minify(raw)
        self.assertLess(len(squeezed), len(raw) * 0.85,
                        "less than 15% off is not worth the risk of doing this at all")

    def test_minifying_twice_changes_nothing(self):
        """A pass that is not idempotent is a pass that is still editing."""
        static = PROJECT_ROOT / "app" / "static"
        once = minify((static / "components.css").read_text(encoding="utf-8"))
        self.assertEqual(once, minify(once))

    def test_every_rule_survives(self):
        """Braces balance and the rule count is unchanged - a cheap check that
        nothing was swallowed wholesale."""
        static = PROJECT_ROOT / "app" / "static"
        for path in sorted(static.glob("*.css")):
            with self.subTest(file=path.name):
                raw = path.read_text(encoding="utf-8")
                squeezed = minify(raw)
                self.assertEqual(squeezed.count("{"), squeezed.count("}"))
                # Comments can contain braces, so compare against a raw copy
                # with its comments removed rather than the original.
                without = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
                self.assertEqual(squeezed.count("{"), without.count("{"))


class WiringTests(unittest.TestCase):
    def test_the_served_bundles_are_squeezed(self):
        from app.main import CSS_BUNDLE_TEXT

        for name, text in CSS_BUNDLE_TEXT.items():
            with self.subTest(bundle=name):
                self.assertNotIn("/* style.css */", text)
                self.assertNotIn("\n\n", text)

    def test_there_is_a_switch_to_turn_it_off(self):
        """A minifier is the kind of thing that breaks one page on one
        browser. The first question then is whether it was the minifier, and
        that should be answerable with a restart rather than a deploy."""
        self.assertIsInstance(SETTINGS.css_minify_enabled, bool)
        main = (PROJECT_ROOT / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("if SETTINGS.css_minify_enabled else", main)


if __name__ == "__main__":
    unittest.main()

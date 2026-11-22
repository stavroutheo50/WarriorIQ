"""Screenshot every main route at phone, tablet and laptop sizes; fail on sideways scrolling.

    python scripts/visual_check.py http://127.0.0.1:8000 --email you@example.com --password-env WIQ_PW
    python scripts/visual_check.py https://warrioriq.eu --out shots/

Written for the site walkthrough of 2026-11-16. For each route and each size
it loads the page, lets the reveal animations settle, saves a full-page
screenshot, and measures document.documentElement.scrollWidth against
clientWidth: a page that is wider than the window scrolls sideways on a phone,
which is the failure this exists to catch. It exits 1 if any page does, and
names the elements that stick out past the right edge.

The password is read from the environment variable named by --password-env
(never from the command line, where it would land in shell history) and is
never printed. Signed-out pages are always checked; signed-in ones only with
credentials. Pages that need an id (/result/<id>) are found from the Fight
library, or passed with --paths.

Needs Playwright with a Chromium: `pip install playwright && playwright install
chromium`, or set CHROMIUM to an existing Chromium executable.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SIZES = {"phone": (390, 844), "tablet": (768, 1024), "laptop": (1366, 768)}
PUBLIC = ["/", "/login", "/signup", "/forgot-password", "/pricing", "/scorer", "/legal", "/terms", "/privacy",
          "/contact", "/cookies", "/this-page-does-not-exist"]
SIGNED_IN = ["/", "/history", "/camp", "/scorer", "/pricing", "/profile", "/settings/privacy", "/settings/billing",
             "/feed", "/coaching", "/gym"]

OVERFLOW_JS = r"""() => {
  const root = document.documentElement, width = root.clientWidth, out = [];
  if (root.scrollWidth > width) {
    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect(), style = getComputedStyle(el);
      if (!r.width || style.position === 'fixed' || r.right <= width + 1) continue;
      const parent = el.parentElement && el.parentElement.getBoundingClientRect();
      if (parent && parent.right > width + 1) continue;          // only the outermost offender
      if (el.closest('.table-wrap, [data-scroll-x]')) continue;  // scrolls inside its own box
      out.push(`${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''}.${String(el.className).split(' ')[0]} right=${Math.round(r.right)}`);
      if (out.length > 8) break;
    }
  }
  return {scrollWidth: root.scrollWidth, clientWidth: width, offenders: out};
}"""


def _settle(page) -> None:
    page.wait_for_load_state("networkidle")
    # Reveal-on-scroll content: scroll through once so it is drawn, then back.
    page.evaluate("""async () => { for (let y = 0; y < document.body.scrollHeight; y += innerHeight) {
        scrollTo(0, y); await new Promise(r => setTimeout(r, 60)); } scrollTo(0, 0); }""")
    page.wait_for_timeout(250)


def _sign_in(context, base: str, email: str, password: str) -> bool:
    page = context.new_page()
    page.goto(f"{base}/login")
    page.fill("#authEmail", email)
    page.fill("#authPassword", password)
    page.click(".auth-submit")
    page.wait_for_load_state("networkidle")
    ok = "/login" not in page.url
    page.close()
    return ok


def _result_paths(context, base: str, limit: int = 1) -> list[str]:
    page = context.new_page()
    page.goto(f"{base}/history")
    links = page.eval_on_selector_all("a[href^='/result/']", "els => els.map(e => e.getAttribute('href'))")
    page.close()
    seen = []
    for link in links:
        path = link.split("?")[0].split("#")[0]
        if path not in seen:
            seen.append(path)
    return seen[:limit]


def check(base: str, paths: list[str], context, out: Path | None, label: str) -> list[str]:
    failures = []
    for size_name, (width, height) in SIZES.items():
        page = context.new_page()
        page.set_viewport_size({"width": width, "height": height})
        for path in paths:
            page.goto(f"{base}{path}")
            _settle(page)
            result = page.evaluate(OVERFLOW_JS)
            bad = result["scrollWidth"] > result["clientWidth"]
            line = f"{'OVERFLOW' if bad else 'ok':8} {label:9} {size_name:6} {path}"
            if bad:
                line += f"  scrollWidth={result['scrollWidth']} clientWidth={result['clientWidth']} " \
                        + "; ".join(result["offenders"])
                failures.append(line)
            print(line, flush=True)
            if out is not None:
                name = re.sub(r"[^a-z0-9]+", "-", f"{label}-{size_name}-{path}".lower()).strip("-")
                page.screenshot(path=str(out / f"{name}.png"), full_page=True)
        page.close()
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--email")
    parser.add_argument("--password-env", default="WARRIORIQ_VISUAL_CHECK_PASSWORD",
                        help="name of the environment variable holding the password")
    parser.add_argument("--paths", default="", help="extra comma-separated paths to check signed in")
    parser.add_argument("--out", type=Path, help="folder for the screenshots (none saved without it)")
    args = parser.parse_args(argv)

    from playwright.sync_api import sync_playwright

    base = args.base_url.rstrip("/")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    with sync_playwright() as playwright:
        # CHROMIUM names a browser to use instead of Playwright's own download.
        executable = os.getenv("CHROMIUM") or None
        browser = playwright.chromium.launch(executable_path=executable)
        guest = browser.new_context()
        failures += check(base, PUBLIC, guest, args.out, "guest")
        guest.close()
        password = os.getenv(args.password_env, "")
        if args.email and password:
            member = browser.new_context()
            if not _sign_in(member, base, args.email, password):
                print("Could not sign in with the given account; signed-in pages not checked.", file=sys.stderr)
                failures.append("sign-in failed")
            else:
                extra = [p for p in args.paths.split(",") if p.strip()]
                failures += check(base, SIGNED_IN + _result_paths(member, base) + extra, member, args.out, "signed-in")
            member.close()
        elif args.email:
            print(f"Set {args.password_env} to check the signed-in pages.", file=sys.stderr)
        browser.close()
    print(f"\n{len(failures)} page(s) with horizontal overflow." if failures else "\nNo horizontal overflow.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

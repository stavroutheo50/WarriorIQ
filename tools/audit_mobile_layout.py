"""Audit pages at phone width: overflow, tap targets under 44px, text under 12px.

    python tools/audit_mobile_layout.py https://warrioriq.eu [--email E --password P] [--paths /a,/b]

Written for the QA pass of 2026-10-04 (item 21), where every page was checked
at 390px. Signed-out pages are always checked; with credentials, the signed-in
pages too (pass --paths for result/select pages, which need a job id). Inline
links inside running text are not counted as tap targets (WCAG 2.5.8). Needs
Playwright and a Chromium; set CHROMIUM to its executable if Playwright cannot
find its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

AUDIT_JS = r"""(() => {
  const vw = innerWidth, out = {overflow: [], small: [], tiny: []};
  const docW = document.documentElement.scrollWidth;
  if (docW > vw) {
    document.querySelectorAll('body *').forEach(e => {const r = e.getBoundingClientRect(); if (r.right > vw + 1 && r.width > 0 && getComputedStyle(e).position !== 'fixed') { const p = e.parentElement && e.parentElement.getBoundingClientRect(); if (!p || p.right <= vw + 1) out.overflow.push(e.tagName.toLowerCase() + '.' + String(e.className).split(' ')[0] + ' r=' + Math.round(r.right)); }});
  }
  const visible = e => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e); return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && !e.closest('[hidden],dialog:not([open]),.sr-only,.visually-hidden'); };
  document.querySelectorAll('a[href],button,input:not([type=hidden]),select,textarea,summary').forEach(e => {
    if (!visible(e)) return;
    if (e.tagName === 'A' && e.closest('p,li:not(.nav li),small,dd,td') && getComputedStyle(e).display === 'inline') return; // inline text links
    let r = e.getBoundingClientRect();
    if ((e.type === 'radio' || e.type === 'checkbox') && e.closest('label')) r = e.closest('label').getBoundingClientRect();
    if (e.type === 'file') return;
    if (r.height < 43.5 || r.width < 43.5) out.small.push(e.tagName.toLowerCase() + (e.id ? '#' + e.id : '.' + String(e.className).split(' ')[0]) + ' "' + (e.textContent || e.value || e.getAttribute('aria-label') || '').trim().slice(0, 24) + '" ' + Math.round(r.width) + 'x' + Math.round(r.height));
  });
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const seen = new Set();
  while (walker.nextNode()) { const t = walker.currentNode; if (!t.textContent.trim()) continue; const el = t.parentElement; if (!el || seen.has(el) || !visible(el) || el.closest('script,style,noscript')) continue; seen.add(el); const fs = parseFloat(getComputedStyle(el).fontSize); if (fs > 0 && fs < 12) out.tiny.push(el.tagName.toLowerCase() + '.' + String(el.className).split(' ')[0] + ' ' + fs + 'px "' + t.textContent.trim().slice(0, 30) + '"'); }
  out.docW = docW; out.small = [...new Set(out.small)]; out.tiny = [...new Set(out.tiny)];
  return out;
})()
"""

SIGNED_OUT = ["/", "/login", "/signup", "/forgot-password", "/pricing", "/analyze", "/privacy", "/terms",
              "/subprocessors", "/contact", "/kickboxing-fight-analysis", "/how-to-record-a-fight-for-analysis",
              "/accessibility", "/ai-transparency", "/legal", "/policies", "/guardian", "/copyright-report"]
SIGNED_IN = ["/analyze", "/analyze/kickboxing", "/history", "/compare", "/camp", "/dashboard", "/coach",
             "/profile", "/settings", "/pricing", "/validation"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("--email")
    parser.add_argument("--password")
    parser.add_argument("--paths", default="")
    parser.add_argument("--width", type=int, default=390)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    problems = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(**({"executable_path": os.environ["CHROMIUM"]} if os.getenv("CHROMIUM") else {}))
        runs = [("out", SIGNED_OUT)]
        if args.email and args.password:
            runs.append(("in", SIGNED_IN + [x for x in args.paths.split(",") if x]))
        for signed, paths in runs:
            page = browser.new_context(viewport={"width": args.width, "height": 844}, is_mobile=True,
                                       has_touch=True).new_page()
            if signed == "in":
                page.goto(args.base + "/login")
                page.fill("input[name=email]", args.email)
                page.fill("input[name=password]", args.password)
                page.click("form[action^='/login'] button[type=submit]")
                page.wait_for_load_state("networkidle")
            for path in paths:
                page.goto(args.base + path)
                page.wait_for_load_state("networkidle")
                found = page.evaluate(AUDIT_JS)
                bad = found["docW"] > args.width or found["small"] or found["tiny"]
                problems += bool(bad)
                print(("FAIL " if bad else "ok   ") + f"{signed} {path}" + (" " + json.dumps(found) if bad else ""))
        browser.close()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

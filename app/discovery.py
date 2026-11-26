"""Being found: a description for AI tools, and telling search engines (2026-11-25).

Owner, 2026-11-25: asked another Claude "what is WarriorIQ" and it could not
find it. The site lets every crawler in (robots.txt, ClaudeBot, GPTBot,
bingbot and Googlebot all get the page), but a new site is not in a search
index until it is submitted or linked to, and the assistants answer from those
indexes (Claude's web search from Brave, ChatGPT's mostly from Bing). This adds
the parts the site itself can do:

- /llms.txt: a plain description of WarriorIQ for language-model tools
  (llmstxt.org), with links to the public pages. It says what WarriorIQ does
  and does not do; no fight's statistics.
- IndexNow (indexnow.org): one request tells Bing, Yandex, Seznam and Naver
  which pages exist, so they crawl them now rather than whenever they find a
  link. The key is generated once and kept beside the database, and served at
  /indexnow-key.txt as the protocol requires.

Search-console ownership tags (Google: WARRIORIQ_SITE_VERIFICATION; Bing:
WARRIORIQ_BING_VERIFICATION) are rendered by base.html.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import urllib.error
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger("warrioriq.discovery")
INDEXNOW_ENDPOINT = "https://api.indexnow.org/indexnow"
KEY_FILE_NAME = "indexnow-key.txt"


def bing_verification_token() -> str:
    return os.getenv("WARRIORIQ_BING_VERIFICATION", "").strip()


def llms_txt(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"""# WarriorIQ

> WarriorIQ is a fight video analysis website for combat sports: kickboxing, boxing, Muay Thai, taekwondo and MMA. A fighter or coach films a sparring session or bout on a phone, uploads it, taps which person is which, and gets a private report on both fighters' movement, pressure, guard and balance, round by round, an estimated score under the sport's rules, and a training plan built from their own numbers.

WarriorIQ runs at {base}/. A free account (Starter) gives one analysis a day; paid plans are listed but billing is not open yet.

What it measures: where each fighter stood and moved, pressure (who walks forward), time in the centre of the ring, guard, balance and, for MMA, time on the ground. Strike counts are shown as estimates and the reports say so. Scores are automatic training estimates, not official judging.

Privacy: fights are private to the account unless shared; uploaded videos are deleted after 30 days unless the uploader chooses to share one to improve WarriorIQ (never for anyone under 18); accounts can export or delete their data at any time.

## Main pages

- [Home]({base}/): what WarriorIQ is and how it works
- [Analyse a fight]({base}/analyze): choose a sport and upload a video
- [Plans]({base}/pricing): Starter (free), Athlete, Athlete Pro and coach and gym plans
- [Ringside Scorer]({base}/scorer): free live round-by-round scoring on a phone, no account needed
- [For coaches]({base}/fight-video-analysis-for-coaches): rosters, assignments and shared reports

## Guides

- [How to record a fight for analysis]({base}/how-to-record-a-fight-for-analysis)
- [Kickboxing fight analysis]({base}/kickboxing-fight-analysis)
- [K-1 fight analysis]({base}/k1-fight-analysis)
- [Muay Thai fight analysis]({base}/muay-thai-fight-analysis)
- [MMA fight analysis]({base}/mma-fight-analysis)
- [Boxing fight analysis]({base}/boxing-fight-analysis)
- [Taekwondo analysis (WT and ITF)]({base}/taekwondo-fight-analysis)
- [AI sparring analysis]({base}/ai-sparring-analysis)

## Trust

- [How WarriorIQ uses AI]({base}/ai-transparency)
- [Privacy policy]({base}/privacy)
- [Terms of service]({base}/terms)
- [Contact]({base}/contact)
"""


def indexnow_key(data_root: Path) -> str:
    """The site's IndexNow key: made once, kept beside the database."""
    path = Path(data_root) / KEY_FILE_NAME
    try:
        if path.exists():
            stored = path.read_text(encoding="utf-8").strip()
            if stored:
                return stored
        key = secrets.token_hex(16)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:                       # another process made it first
            return path.read_text(encoding="utf-8").strip()
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(key)
        return key
    except OSError:
        LOGGER.exception("indexnow_key_unavailable")
        return ""


def submit_to_indexnow(base_url: str, paths: tuple[str, ...], key: str, *,
                       opener=urllib.request.urlopen) -> dict:
    """Tell IndexNow's search engines about these pages. Returns {"ok", "status", "count"}."""
    base = base_url.rstrip("/")
    host = base.split("://", 1)[-1].split("/", 1)[0]
    if not base.startswith("https://") or not key:
        return {"ok": False, "status": "not_configured", "count": 0}
    body = json.dumps({"host": host, "key": key, "keyLocation": f"{base}/{KEY_FILE_NAME}",
                       "urlList": [f"{base}{path}" for path in paths]}).encode("utf-8")
    request = urllib.request.Request(INDEXNOW_ENDPOINT, data=body, method="POST",
                                     headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with opener(request, timeout=15) as response:
            status = int(getattr(response, "status", 200))
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
    except (urllib.error.URLError, OSError) as exc:
        LOGGER.warning("indexnow_unreachable error=%s", type(exc).__name__)
        return {"ok": False, "status": "unreachable", "count": 0}
    return {"ok": status in {200, 202}, "status": status, "count": len(paths)}

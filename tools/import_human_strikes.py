"""Turn a person's strike labels from the labelling page into answer keys.

The labelling page (a private claude.ai page the owner shares with a paid
labeller, 2026-11-10) plays each densely labelled clip's covered stretch and
saves one document per clip and labeller:

    {"clip": "S04", "labeller": "<opaque id>", "marks": [{"t": 1.23, "fighter": "A",
     "family": "punch", "sure": true}], "unreadable": [[a, b]], "watched": 42.1,
     "finished": true, "updated": "..."}

with `t` in seconds from the start of the clip, A the fighter boxed blue on the
page's still and B the one boxed red (the analyser's fighter_A and fighter_B
at the clip's first frame). This writes strikes.human.json beside the clip's
strikes.json, in the same layout, which tools/benchmark_dense_strikes.py and
tools/fit_strike_check.py then use in place of Claude's labels.

    python tools/import_human_strikes.py labels.json            # write the keys
    python tools/import_human_strikes.py labels.json --compare  # also: how Claude's labels did

labels.json is the page's `labels` collection: a JSON list of documents, or
one JSON object per line, each either the document itself or {"data": ...}.
Only the stretch the person watched is kept (all of it once they pressed
"Clip finished"). With several labellers on one clip, the one who watched the
most is used; the others are listed in the notes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import benchmark_dense_strikes as dense  # noqa: E402

FAMILIES = {"punch", "kick", "knee"}
# The page shows one clip for a stretch some folders share: S04K and S07K are
# S04 and S07 re-tracked for kicks, the same video and the same strikes.
ALSO = {"S04": ["S04K"], "S07": ["S07K"]}


def read_documents(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    try:
        loaded = json.loads(text)
        rows = loaded if isinstance(loaded, list) else [loaded]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]
    docs = []
    for row in rows:
        doc = row.get("data", row) if isinstance(row, dict) else None
        if isinstance(doc, dict) and doc.get("clip"):
            docs.append(doc)
    return docs


def folders() -> dict[str, Path]:
    return {fight.name: fight for fight in dense.dense_fights()}


def pick(docs: list[dict]) -> dict[str, tuple[dict, list[dict]]]:
    """Per clip: the labeller who watched the most, and the others."""
    by_clip: dict[str, list[dict]] = {}
    for doc in docs:
        by_clip.setdefault(str(doc["clip"]), []).append(doc)
    chosen = {}
    for clip, rows in by_clip.items():
        rows.sort(key=lambda d: (bool(d.get("finished")), float(d.get("watched") or 0)), reverse=True)
        chosen[clip] = (rows[0], rows[1:])
    return chosen


def answer_key(doc: dict, claude: dict, others: list[dict]) -> dict:
    """The person's marks in strikes.json's layout and the video's own time."""
    start, end = (float(x) for x in claude["covered"])
    watched = (end - start) if doc.get("finished") else min(end - start, float(doc.get("watched") or 0))
    strikes = []
    for mark in doc.get("marks") or []:
        t, fighter, family = float(mark.get("t", -1)), str(mark.get("fighter", "")), mark.get("family")
        if 0 <= t <= watched and fighter in {"A", "B"} and family in FAMILIES:
            strikes.append({"t": round(start + t, 3), "fighter": fighter, "family": family,
                            "outcome": "unclear", "sure": bool(mark.get("sure", True))})
    unreadable = [[round(start + float(a), 2), round(start + float(b), 2)]
                  for a, b in (doc.get("unreadable") or []) if 0 <= float(a) < float(b)]
    return {
        "clip": doc["clip"],
        "covered": [start, round(start + watched, 2)],
        "unreadable": unreadable,
        "strikes": sorted(strikes, key=lambda s: (s["t"], s["fighter"])),
        "notes": (f"Labelled by a person on the WarriorIQ labelling page (labeller {doc.get('labeller')}, "
                  f"saved {doc.get('updated')}), watching the video. A is the fighter boxed blue on the "
                  f"page's still, B the one boxed red. Outcomes were not asked."
                  + (f" Other labellers on this clip, not used: {[d.get('labeller') for d in others]}." if others else "")),
    }


def agreement(claude: dict, human: dict) -> dict:
    """How Claude's labels did against the person's, in the person's watched stretch."""
    start, end = human["covered"]
    spans = human.get("unreadable") or []
    inside = lambda s: start <= s["t"] <= end and not dense._inside(s["t"], spans)  # noqa: E731
    mine = [s for s in claude["strikes"] if inside(s) and s.get("sure", True)]
    truth = [s for s in human["strikes"] if inside(s) and s.get("sure", True)]
    pairs = sorted((abs(a["t"] - b["t"]), i, j) for i, a in enumerate(mine) for j, b in enumerate(truth)
                   if a["fighter"] == b["fighter"] and abs(a["t"] - b["t"]) <= dense.MATCH_SECONDS)
    used_a, used_b = set(), set()
    for _, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return {"person": len(truth), "claude": len(mine), "agreed": len(used_a)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("labels", type=Path, help="the page's labels collection, exported as JSON")
    parser.add_argument("--compare", action="store_true", help="also print how Claude's labels agreed")
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = parser.parse_args(argv)
    known = folders()
    for clip, (doc, others) in sorted(pick(read_documents(args.labels)).items()):
        if clip not in known:
            print(f"{clip}: no labelled clip folder here, skipped")
            continue
        claude = json.loads((known[clip] / "strikes.json").read_text(encoding="utf-8"))
        key = answer_key(doc, claude, others)
        if key["covered"][1] <= key["covered"][0]:
            print(f"{clip}: nothing watched yet, skipped")
            continue
        line = f"{clip}: {len(key['strikes'])} strikes over {key['covered'][1] - key['covered'][0]:.1f} s"
        if args.compare:
            a = agreement(claude, key)
            line += f"; Claude's labels: {a['agreed']} of the person's {a['person']} found, {a['claude'] - a['agreed']} extra"
        print(line + ("" if doc.get("finished") else " (not finished: watched part only)"))
        if not args.dry_run:
            for name in [clip, *ALSO.get(clip, [])]:
                if name in known:
                    (known[name] / dense.HUMAN_LABELS).write_text(json.dumps(key, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download the public combat-sport datasets WarriorIQ trains and measures on.

Private use only: nothing fetched here is served, shown or committed. Every
source is written down with its licence in ``PROVENANCE.json`` beside the
data, so a source whose terms change - or that turns out not to be usable in
a paid product - can be found and dropped, along with every sequence that came
from it (the importers stamp ``fight_id`` with the source prefix).

Searched 2026-10-04. What exists, and what it is good for:

    source            sport           what is labelled                         licence
    tkd_kick3         taekwondo       765 kicks: front, roundhouse, axe        CC BY 4.0
    boxingvi          boxing          6,915 punches, 6 types, keypoints        none stated
    strikemetrics     kickboxing/MT   ~400 strikes in 5 pro fights; F1/F2/Ref  MIT (annotations)
                                      boxes. Sparse: not every strike marked
    fight_judge       MMA             5,106 UFC frames, fighter boxes + pose   CC BY-NC-SA 4.0
    olympic_boxing    boxing          2,278 clips: head/body/block/miss per    non-commercial
                                      hand (needs a Kaggle token)

Nothing public was found with complete strike labels for full kickboxing,
Muay Thai or MMA fights, nor for elbows, spinning or jumping kicks, takedowns,
or WT/ITF taekwondo competition scoring.

Usage:
    python tools/fetch_public_datasets.py                 # everything reachable
    python tools/fetch_public_datasets.py --only tkd_kick3 boxingvi

boxingvi's data is on Google Drive and needs ``pip install gdown``;
olympic_boxing needs the Kaggle CLI and KAGGLE_USERNAME / KAGGLE_KEY in the
environment. Without them those sources are skipped with the reason printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "dataset" / "public"


@dataclass(frozen=True)
class Source:
    name: str
    sport: str
    labels: str
    licence: str
    commercial_use: str          # "yes", "no", or "unclear"
    homepage: str
    kind: str                    # "zenodo", "git", "gdrive" or "kaggle"
    location: str


SOURCES = (
    Source("tkd_kick3", "taekwondo", "front, roundhouse and axe kicks as keypoint sequences",
           "CC BY 4.0", "yes", "https://zenodo.org/records/20390892",
           "zenodo", "https://zenodo.org/records/20390892/files/TKD-Kick3.zip?download=1"),
    Source("boxingvi", "boxing", "punch type and start/end frame, AlphaPose keypoints",
           "none stated", "unclear", "https://github.com/Bikudebug/BoxingVI",
           "gdrive", "https://drive.google.com/drive/folders/1Vyl8twJQ1qkqEPwhvfsrJsJ8nLQ92uoy"),
    Source("strikemetrics", "kickboxing, muay thai", "strike type at one frame; Fighter 1, Fighter 2 and "
           "referee boxes; MoveNet keypoints for every frame. Not every strike is marked",
           "MIT", "yes", "https://github.com/sswhitehat/StrikeMetrics---Kickboxing-AI-Tool",
           "git", "https://github.com/sswhitehat/StrikeMetrics---Kickboxing-AI-Tool"),
    Source("fight_judge", "mma", "fighter boxes and 17 keypoints on UFC frames",
           "CC BY-NC-SA 4.0 (data), MIT (code)", "no", "https://github.com/hasanfaesal/fight-judge",
           "git", "https://github.com/hasanfaesal/fight-judge"),
    Source("olympic_boxing", "boxing", "head / body / block / miss for each hand, by licensed referees",
           "non-commercial", "no",
           "https://www.kaggle.com/datasets/piotrstefaskiue/olympic-boxing-punch-classification-video-dataset",
           "kaggle", "piotrstefaskiue/olympic-boxing-punch-classification-video-dataset"),
)


def _fetch(source: Source, target: Path) -> str | None:
    """Download one source into ``target``. Returns why it was skipped, or None."""
    if source.kind == "zenodo":
        target.mkdir(parents=True, exist_ok=True)
        archive = target / Path(source.location.split("?")[0]).name
        if not archive.exists():
            with urllib.request.urlopen(source.location, timeout=600) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
        return None
    if source.kind == "git":
        if not (target / ".git").exists():
            subprocess.run(["git", "clone", "--depth", "1", source.location, str(target)], check=True)
        return None
    if source.kind == "gdrive":
        try:
            import gdown  # noqa: PLC0415 - optional, only this source needs it
        except ImportError:
            return "needs `pip install gdown` (Google Drive download)"
        if not target.exists():
            gdown.download_folder(url=source.location, output=str(target), quiet=True)
        return None
    if source.kind == "kaggle":
        if not shutil.which("kaggle") or not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
            return "needs the Kaggle CLI and KAGGLE_USERNAME / KAGGLE_KEY"
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", source.location, "-p", str(target), "--unzip"],
                       check=True)
        return None
    raise ValueError(f"unknown source kind {source.kind!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", choices=[s.name for s in SOURCES])
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    provenance_path = args.out / "PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    for source in SOURCES:
        if args.only and source.name not in args.only:
            continue
        print(f"{source.name}: {source.licence} (commercial use: {source.commercial_use})")
        try:
            skipped = _fetch(source, args.out / source.name)
        except (OSError, subprocess.CalledProcessError) as exc:
            skipped = f"download failed: {type(exc).__name__}: {exc}"
        if skipped:
            print(f"  skipped - {skipped}")
            continue
        provenance[source.name] = {**asdict(source),
                                   "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        print(f"  -> {args.out / source.name}")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

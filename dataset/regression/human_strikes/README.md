# Strikes marked by a person on the labelling page

`2026-11-10.labels.json` is the labelling page's `labels` collection as the
owner left it on 2026-11-10: 9 of the 14 clips, 112 strikes, each clip marked
finished. The labeller id is the page's opaque per-person id, nothing else.

Rebuild the answer keys (strikes.human.json beside each clip's strikes.json)
on any machine that has the clips:

    python tools/import_human_strikes.py dataset/regression/human_strikes/2026-11-10.labels.json --compare

What it showed on 2026-11-10, against these 9 clips:

- Claude's own labels found 41 of the person's 111 strikes and added 116
  the person did not mark, so they are not an answer key.
- The strike count (tools/benchmark_dense_strikes.py, 0.35 s window):
  precision 29%, recall 35%. With a 0.5 s window, 35% and 43%. Taps are
  centred on the counted moment (median offset -0.03 s) with a spread of
  about +/-0.5 s, so the 0.35 s window costs some real matches.
- The video punch check separates real from false punches only weakly on
  these clips (held-out AUC 0.25-0.71), so it was not refitted on them.

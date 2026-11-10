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

Per strike against per fighter (2026-11-10, same 9 clips, 0.5 s window):

- Individual counted moments: 51 of 136 are a strike the person marked
  (38%); the counter finds 51 of the person's 111 (46%).
- Each fighter's total on a clip: correlation 0.83 with the person's,
  median error 2 strikes; the busier fighter is right on 6 of the 8 clips
  where the two differ. The errors largely cancel in a total.
- 25 of the 85 false moments fall on a strike by the other fighter (a
  defender's reaction counted as a punch). Keeping one of each such pair
  (by contact distance) lifts per-moment precision only from 38% to 42% and
  makes the totals slightly worse (correlation 0.76), so it was not adopted.
- No single measured feature separates real from false counts well (best:
  contact distance, AUC 0.64). More rule tuning will not fix this; a learned
  model needs far more labelled footage than 9 clips.

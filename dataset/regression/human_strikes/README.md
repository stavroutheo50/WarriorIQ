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

## 2026-11-24: six stock clips, a person's labels (tools/strike_errors.py)

54 sure strikes on dataset/stock_strikes/mixkit_* (strikes.human.json).

- Analysis PC tracks: 63 counted, 27 right (precision 43%, recall 50%).
  Misses: 7 proposed for the other fighter, 7 off in time, 6 combos counted
  once, 5 dropped by the attempt gate, 1 pair rule, 1 never proposed. False:
  15 nothing marked (10 on mixkit_40968: 11 counted, 2 marked), 11 off in
  time, 8 the other fighter's strike, 2 double counts.
- Rebuilt here on CPU (yolo26m-pose), 12 fps: 48 counted, 21 right. At the
  clips' full 24 fps: 68 counted, 30 right - recall 39% -> 56%, precision
  unchanged at 44%, and false "other fighter" counts 12 -> 16. Not adopted:
  twice the pose work for more counts that are still wrong half the time.
- Telling a real punch from the defender's flinch, over 12 skeleton measures
  (torso, head and wrist travel toward or away from the opponent, elbow
  angle, closing distance, speed, contact distance): best AUC 0.64 at 12 fps,
  0.62 at 24 fps; the direction measures 0.47-0.57. No rule was built on them.
- "Who threw more" from the counts (a clear gap of 3 and 25%): right on 0 of
  2 clips where it would claim one; any difference at all: 2 of 5. Not shown.

The skeleton rules are at their limit for strikes. The next step is a learned
model on many labelled real strikes: the owners' quick-check answers
(app/accuracy.py), once there are a few hundred.

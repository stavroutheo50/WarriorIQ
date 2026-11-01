# Labelled own sparring

Sparring filmed by WarriorIQ's own people, every strike labelled by hand for
`tools/benchmark_dense_strikes.py`, in the same format as
`dataset/stock_strikes/`.

Each folder has:

- `strikes.json`: the strikes (fighter as tracked, family, time of fullest
  extension, outcome where clear, sure or not), what was covered, the spans
  that could not be judged (clinches, people crossing the camera, fighters
  out of frame) and notes, including any A/B swap the analysis made.
- `source.json`: who filmed it, when, the ruleset, the frame the two fighters
  were picked on, and that everyone in it is an adult who agreed to filming
  and to its use for improving WarriorIQ.
- `track.jsonl.gz`: the analysis's pose track. **Never committed**, and
  neither is the video: they stay on the machine that filmed them.

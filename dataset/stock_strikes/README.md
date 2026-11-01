# Labelled stock clips

Free-licence stock fight clips, every strike labelled by hand so
`tools/benchmark_dense_strikes.py` measures both false strikes and missed ones.

Each folder has:

- `strikes.json`: the strikes, what was covered, the spans that could not be
  judged, and notes (including any fighter swap the analysis made).
- `source.json`: where the clip comes from, its licence, the ruleset and the
  frame the two fighters were picked on.
- `track.jsonl.gz`: the analysis's pose track. **Not committed**: pose data of
  identifiable athletes stays out of this public repository. Rebuild it with
  `python tools/make_stock_strikes.py`.

These are short, staged clips filmed by a stock-video crew, not phone footage
of real sparring. They are a start, not a substitute for labelled fights from
WarriorIQ's own users. Each clip's licence was checked on the date in
`source.json`; check it again before using a clip anywhere public.

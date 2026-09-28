# Identity benchmark: two pankration bouts

Who is really fighter A and who is fighter B, marked by eye on frames of two
real amateur pankration bouts (2008 California Pankration Championships),
filmed handheld in a busy school gym. Scored by `tools/identity_benchmark.py`.

The videos are not stored here. They are published by Subfighter.com on the
Internet Archive under **Creative Commons Attribution 2.0**:

- `ma604.json`: https://archive.org/details/Subfightercom-BobbyMoralesAge162008CaliforniaPankrationChampionshipsMa604
- `ma640.json`: https://archive.org/details/Subfightercom-BobbyMoralesAge162008CaliforniaPankrationChampionshipsMa640

Each file gives the analysis window and the starting boxes used, what each
fighter and the referee wear, and for each marked frame:

- `phase: "standing"`: `A` and `B` are the real fighters' boxes (from the pose
  detector, chosen by a person). `A` is null when that fighter was hidden.
- `phase: "ground"`: the detector saw both fighters as one box (`pair`), so
  no analysis can tell them apart there; scored separately.

Frames during breaks and after the bout are not marked.

Why these bouts: the referee and one fighter both wear black shirts, the
camera pans constantly, other bouts fight in the background and spectators
sit at the mat edge. On the analysis as it stood when they were marked,
fighter A's box was on fighter A in 4 of 20 standing frames of ma640.

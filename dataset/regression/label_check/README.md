# Label checks

A person's check of a random sample of Claude's strike and ground labels.

`2026-11-08.json` is the key for the WarriorIQ Label Check page: one entry per
clip shown, with the clip id, the source fight, the labelled time (strikes: `t`,
fighter, family; ground: `second` and its code) and whether it was labelled
ground. Strike clips are 1.2 s around the labelled moment, ground clips the
labelled second. The answers given on the page are matched to it by id.

Sample: 5 sure strikes from each of S01, S04, S07, SERI, V16, V17, V18 and VID,
and up to 3 ground and 3 standing seconds from each pankration bout in
`../grappling_pankration`, random seed 20261108.

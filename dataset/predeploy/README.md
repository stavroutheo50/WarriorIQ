# Pre-deploy regression clips

Run before every deploy, on the GPU worker, from the commit being deployed:

    python tools/predeploy_regression.py dataset/predeploy/manifest.json

It analyses each clip with the real pipeline and fails if any clip breaks a
product rule: the whole video must be analysed; there must be no kick or knee
without visible legs; analysis must take no longer than the video; and the
report must give one verdict. See `tools/predeploy_regression.py`.

The clips are not committed (they show real people). For each entry in
`manifest.json`, put the file here and fill in the boxes and selection time.

| id | what it must be |
|---|---|
| kickboxing, boxing, muay_thai, taekwondo, mma | 30-120 s of a real two-fighter bout of that sport, both fighters full-body, with legs in shot |
| solo | one person shadowboxing or on a bag or pads, 20-60 s |
| waist_up | two boxers framed from the waist up, legs never in shot (like the K-1 clip QA used) |
| rotated | a phone clip filmed sideways **without** a rotation tag; set `expected_turn` (clockwise degrees needed) |
| no_people | a clip with nobody in it (empty ring, title card) |
| corrupt | a file that is not a decodable video (a truncated upload is ideal) |

`no_people` and `corrupt` can be made from scratch. The others need real
footage you have the rights to use.

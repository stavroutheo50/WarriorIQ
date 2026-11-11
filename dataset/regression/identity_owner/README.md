# The owner's own fights: who is who, marked by eye

Three of the owner's real bouts (2026-11-10), added because the owner saw the
fighters mixed up and lost in the skeleton replay of a fight of his, and the
existing identity clips (public-domain phone sparring, pankration) had no
seated coach beside the mat, which is what these have:

- `f5736.json`: WAKO broadcast, one mat; coaches seated at the edge of the mat.
- `fb883.json`: phone recording of a multi-mat hall, 13-18 people in shot; the
  pair on the near mat.
- `f1.json`: broadcast, one mat; a coach seated beside the mat all bout, the
  most visible person in the picture.

The videos are not stored in Git. Each file lists where the video may be on a
machine (`video`): `dataset/public/owner_fights/` or the analysis PC's
`fights/` folder. A machine without them skips these fights.

Same layout as `../identity_pankration`, starting at the frame
`tools/verified_seeds.json` confirms (where both fighters are separated from
the coaches and the referee). One frame every 4 seconds; on each, every person
was detected (yolo26m-pose, imgsz 1280, conf 0.12, because the fighters are
50-70 pixels tall) and A and B were picked by kit colour on a numbered sheet.
`null` means that fighter was hidden, merged with someone else, or not
detected on that frame; those frames are not scored for that fighter.

f1 has a variable frame rate. Its frames were read the way the analysis
reads them (one seek to the first marked frame, then forward), because a seek
straight to each frame lands on a different picture in such a file: the
first version of f1.json was marked that way and scored the analysis 7 of 15
when it was really 15 of 17.

Replayed from the analysis PC's recording (identity-recording-20261110-1813)
through core/identity.py on 2026-11-11: f1 15/17, f5736 31/31, fb883 20/21
right, no swaps and no box on a coach, referee or spectator; the misses are
frames with no box at all.
Marked by Claude from the sheets; identity by kit colour (blue against red or
black) is an easy question, unlike judging strikes.

Scored by `tools/run_identity_benchmark.py` and recorded for replay by
`tools/identity_recording.py record` / `tools/share_identity_recording.ps1`,
which both pick these files up.

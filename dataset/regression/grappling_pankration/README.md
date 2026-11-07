# Pankration bouts: standing or on the ground, second by second

Five amateur pankration (MMA-style) bouts from archive.org's Subfighter.com
uploads, each licensed **CC BY 2.0** (attribution: Subfighter.com, via
archive.org; item names in each JSON). The videos are not stored here:
download the item's MP4 into `dataset/public/grappling_clips/<name>.mp4`.

Every second of each video is labelled from one frame per second:

| code | meaning |
|------|---------|
| S  | standing apart |
| C  | standing clinch, or a takedown in progress |
| GA | on the ground, fighter A on top |
| GB | on the ground, fighter B on top |
| GN | on the ground, nobody clearly on top (scramble, rolling, hidden) |
| X  | not judged: hidden by a spectator, between rounds, or not the bout |

Fighter A and B are described in each file's `fighters` line. Labelled by
Claude from contact sheets and not yet checked by a person; treat a
single second as approximate and whole stretches as reliable.

Used to build and measure MMA ground detection. Nothing reaches a report
until it matches these labels.

# Handheld phone sparring: who is who, labelled by eye

Four clips from the public-domain archive.org item
[`boxingsparring`](https://archive.org/details/boxingsparring)
(Public Domain Mark 1.0). They are WhatsApp phone recordings of outdoor boxing
sparring: 848x480, a handheld camera that pans and zooms, and in V18 a coach
walking through the shot. The videos are not stored here; download them by
the `file` name in each JSON.

Each JSON uses the same layout as `../identity_pankration`. There is one frame
every 4 seconds from 2 s to 122 s. On each frame every person was detected,
and A and B were checked by eye on a drawn sheet. `null` means that fighter
was hidden or not detected on that frame.

Measured 2026-09-29, first ~55 s of each clip replayed through the identity
manager (frames scored against these labels, IoU >= 0.5):

|                     | right | swapped | other | missing |
|---------------------|------:|--------:|------:|--------:|
| before swap correction | 60 | 19 | 3 | 22 |
| with swap correction   | 73 |  5 | 3 | 23 |

See `SETTINGS.swap_correction_margin` and
`SETTINGS.max_suspicious_handoffs_per_minute` in `core/config.py`.

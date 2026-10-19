# Audit fixes — audit of 14 October 2026

Each of the 55 issues: what changed, where, and how it was checked. One commit
per phase on `claude/jolly-brahmagupta-392knu`. Each phase has a test file,
`tests/test_audit_2026_10_14_phase<N>.py`.

"Suite" below means the full test run, `python -m pytest -q -n 4`. After the
last phase it gave **2267 passed, 29 skipped, 0 failed**.

Some fixes need the owner, money, hardware or a lawyer before they are
complete. Those are listed in [OPS_TODO.md](OPS_TODO.md), and the rows
below point to it.

## Phase 1 — Analysis speed (`00d33e7`)

| # | Change | Files | Verified |
|---|---|---|---|
| 1 | **Pose and timing:**<br>• Pose runs in batches (`WARRIORIQ_POSE_BATCH=8`, falling back to per-frame); the TensorRT engine is built for that batch and cached per GPU.<br>• Each analysis logs `analysis_stage_seconds` and its realtime ratio.<br>• The pose stride was already governed by measured throughput, with skipped frames filled in.<br><br>Fights are analysed on WarriorIQ's own GPU PC (`worker.py`); no cloud GPU service is used. A cloud worker added in this phase was removed afterwards at the owner's request. | `deploy/release.py`, `core/trt_engine.py`, `core/pose_tracker.py`, `core/backtrack.py`, `core/analyzer.py`, `core/config.py`, `tools/benchmark_realtime.py` | Unit tests; benchmark script (see Final checks). **There is no GPU in this environment, so the ratio of 1.0 or less was not measured.** See OPS_TODO. |
| 21 | The live page shows one stage at a time: its label, its own percentage and its own "analysed through" line. | `core/progress_view.py`, `core/types.py`, `app/main.py`, `app/templates/progress.html` | Phase 1 tests; checked in the browser on a running analysis |
| 22 | Time left is estimated per stage from the measured speed after about 50 frames. | `core/progress_eta.py`, `core/analyzer.py` | Phase 1 tests |
| 23 | One release string (`core.build_info.release_version`) on `/health` and in `deploy/release.py`.<br><br>When the worker and site versions differ:<br>• the owner gets one alert;<br>• the admin page shows a flag;<br>• users see "Analysis is paused for an update", with no timer. | `core/build_info.py`, `core/worker_alerts.py`, `app/main.py`, `app/templates/admin.html` | Phase 1 tests |

## Phase 2 — Fighter identity (`d3bef4e`)

| # | Change | Files | Verified |
|---|---|---|---|
| 2 | **Re-entry:** a fighter out of sight for over a second is re-identified by appearance (pooled ReID against the tapped frame, or the kit histogram), with a stricter threshold.<br><br>**Officials:** tracks the pooled referee probability marks as an official become a third-person class, never A or B.<br><br>**Camera movement:** it is measured on the background (`core/camera_motion.py`) and the remembered boxes are moved with it before matching.<br><br>**Merged boxes:** when one fighter's box swallows the other's, SAM2 is asked for at once.<br><br>**Partial coverage:**<br>• A fighter followed for 20–45% of the fight is reported at low confidence instead of failing.<br>• The score covers only rounds where both fighters were followed. | `core/identity.py`, `core/camera_motion.py`, `core/analyzer.py`, `core/report.py`, `core/config.py` | Phase 2 tests (synthetic re-entry, camera shift, official tracks, partial coverage). **The real-footage failure rate was not re-measured.** See OPS_TODO. |
| 11 | Only nearly identical boxes (IoU > 0.6) are refused. A hand-drawn box or two separate detections are never refused; at most a warning is shown. | `core/selection_check.py`, `app/templates/select.html`, `app/main.py` | Phase 2 tests. Final check: two heavily overlapping boxes (IoU 0.79) with one drawn by hand give `warn`, not a block. |
| 12 | A tap selects the smallest box containing it; ties go to the nearest centre. | `app/templates/select.html`, `core/selection_check.py` | Phase 2 tests |
| 13 | When there is no clear start, the frame with two confident, least-overlapping people is chosen. | `core/person_detect.py` | Phase 2 tests |
| 24 | The failed report names the cause and the matching fix (steady the camera / keep both in shot and trim / film closer / re-tap). | `core/report.py`, `app/templates/result.html` | Phase 2 tests |
| 41 | Fight Camp offers "Re-tag and re-run" on each fixable failed fight. | `app/templates/camp.html`, `app/main.py` | Phase 2 tests |

## Phase 3 — Server performance (`8a61066`)

| # | Change | Files | Verified |
|---|---|---|---|
| 3 | `run.py` starts `WEB_CONCURRENCY` processes (4 on Render). SQLite was already WAL with a busy_timeout. The session secret is created exclusively, so all processes share one key. | `run.py`, `render.yaml`, `app/main.py` | 4 local processes: 8 parallel `/terms` in 0.03–0.14 s; sign-in and CSRF work across processes |
| 4 | • Signed-out visitors no longer trigger a job-file scan.<br>• The schema script runs once per process.<br>• New indexes on fights, sessions, points_ledger and camp_missions.<br>• Static public pages send `Cache-Control: private, max-age=300` with `Vary: Cookie`.<br>• The page-view flush runs off the event loop. | `app/main.py`, `core/db.py` | Cold burst of 8 `/terms` went from 1.47 s to 0.28 s; Phase 3 tests |
| 25 | • `/media` answers range requests (206, ETag).<br>• An MP4 with its index at the end is remuxed with faststart on upload.<br>• The replay preloads the video.<br>• Optional signed CDN URLs. | `app/main.py`, `core/video.py`, `core/config.py`, `app/templates/replay.html` | Range request returned 206; Phase 3 tests. CDN setup is in OPS_TODO. |

## Phase 4 — Metrics honesty (`0a47e0b`)

| # | Change | Files | Verified |
|---|---|---|---|
| 5 | Guard is wrist-to-chin distance over shoulder width. Both wrists are required (the farther one decides), readings are smoothed and taken only from confident keypoints. It is labelled "Experimental" everywhere. | `core/guard.py`, `core/metric_catalog.py`, templates | Synthetic-pose unit tests. The figure varies by up to 0.067 across sampling strides; this is documented. |
| 6 | Fight Camp no longer says the figures "are real". | `app/templates/camp.html` | Phase 4 tests |
| 7 | A "Why this score" explanation (landed strikes, the ruleset's weights, round by round). No score is shown while strike numbers are withheld. | `core/scoring.py`, `app/templates/result.html` | Phase 4 tests |
| 8 | One rule decides whether strike numbers are shown (`core.report.strike_numbers_shown`), used by the report, replay, Compare and Fight Camp. | `core/report.py`, `app/main.py`, templates | Phase 4 tests |
| 28 | Targets are set only for reliable metrics that have a reference range. Otherwise the plan says why there is no number target. | `core/metric_catalog.py`, `core/coaching.py`, `core/camp.py` | Phase 4 tests |
| 35 | Compare calls a difference "level" by one rule based on measurement noise, for every metric. | `core/squad.py`, `core/coaching.py` | Phase 4 tests |
| 38 | Reworded to "Fighters told apart only N% of the time". | `app/templates/camp.html` | Phase 4 tests |
| 39 | Changes are shown in percentage points ("pp"). | `core/progress_insights.py`, templates | Phase 4 tests |
| 40 | Ring centre change gets a unit ("on the 0–100 scale"). | `core/progress_insights.py` | Phase 4 tests |

## Phase 5 — Upload and selection (`2628a95`)

| # | Change | Files | Verified |
|---|---|---|---|
| 14 | The Start panel follows the A and B cards and is no longer sticky. | `app/static/fighter-selection.css`, `app/templates/select.html` | Browser at 1280, 1440 and 390 px |
| 15 | "More options" spans the full width. | `app/static/fighter-selection.css` | Browser; Phase 5 tests |
| 16 | The profile's main fighter is pre-selected. | `app/main.py`, `app/templates/analyze.html` | Phase 5 tests |
| 17 | • The browser rejects files under 50 KB and files without a video signature.<br>• The server checks the real file type with ffprobe, falling back to ffmpeg. | `app/templates/analyze.html`, `core/video.py`, `app/main.py` | Phase 5 tests, including a forged 12-byte `ftyp` file |
| 18 | The toast that covered the fighter dropdown is removed. | `app/templates/analyze.html` | Phase 5 tests |
| 19 | Kicks or knees found in a boxing upload trigger a suggestion to switch sport, never a block. | `core/sport_check.py`, `app/templates/result.html` | Phase 5 tests |
| 20 | Trim (from / to) on the selection page; the tapped frame must lie inside the kept part. | `app/templates/select.html`, `app/main.py` | Phase 5 tests |

## Phase 6 — Report, replay, library, Compare, Camp (`215eb59`)

| # | Change | Files | Verified |
|---|---|---|---|
| 26 | A replay chapter scrolls the video into view. | `app/templates/replay.html` | Browser; Phase 6 tests |
| 27 | Both skeletons are drawn; a fighter not tracked at that moment gets a "not tracked here" marker. | `app/templates/replay.html` | Browser; Phase 6 tests |
| 29 | Exactly one H1 on the Compare, report and replay pages. | templates, CSS | Phase 6 tests |
| 30 | Optional event and opponent names (nullable columns, explicit migration). Titles are the event name, else "Sport · Date". | `core/db.py`, `core/squad.py`, `app/main.py`, `history.html` | Phase 6 tests, including a schema compatibility test |
| 31 | The fighters' names are used everywhere, including who is ahead. | `app/main.py`, `history.html` | Phase 6 tests |
| 33 | Library filters (status, sport, ruleset) and bulk delete of the owner's own fights. | `app/main.py`, `history.html` | Browser; Phase 6 tests (owner-only) |
| 34 | Fights that failed the identity check are marked and not compared by default. | `app/main.py`, `compare.html` | Phase 6 tests |
| 36 | `/compare?a=..&b=..#results` gives a comparison its own address, which opens only for the owner. | `app/main.py`, `compare.html` | Phase 6 tests |
| 37 | The Compare picker shows a picture, title, date, ruleset, score and status. | `compare.html` | Browser; Phase 6 tests |
| 42 | `/dashboard` is a 301 redirect to `/camp`; every link now points to `/camp`. | `app/main.py`, templates | Phase 6 tests |

## Phase 7 — Navigation, account, global (`0e69b62`)

| # | Change | Files | Verified |
|---|---|---|---|
| 43 | GET `/analyze/<sport>` no longer saves the sport. Choosing a sport is POST `/sport` (CSRF-protected); uploading a fight also saves its sport. | `app/main.py`, `base.html` | Phase 7 tests (no Set-Cookie on GET; POST with a safe `next`) |
| 44 | The avatar letter and label come from the display name, falling back to the email while the name is still the placeholder. | `app/main.py`, `core/db.py`, `base.html` | Phase 7 tests |
| 45 | The account menu gains Settings, Plans and billing, Help and Privacy; Fight library is removed from it. | `base.html` | Browser (menu items listed); Phase 7 tests |
| 46 | The sport switcher marks the current sport with a ✓ and `aria-current`. | `base.html`, `system.css` | Phase 7 tests |
| 47 | The header collapses to the menu below 1024 px. | `fixes.css`, `product.css` | Browser at 980, 1023, 1024 and 1280 px: one 68 px line throughout, menu button below 1024 |
| 48 | The copyright line stays on one line. | `base.html`, `system.css` | Browser at 375–1440 px: one line at 18.6 px tall |
| 49 | Signed out, `/result`, `/replay` and `/progress` redirect to `/login?next=` for every id. Signed in, a missing fight or someone else's is still 404. | `app/main.py` | Phase 7 tests compare the responses for an existing and a missing id; final check |
| 50 | GitHub and Google are equal-size buttons on `/login` (GitHub stays sign-in only). | `auth.html`, `system.css` | Browser: both 393×46 px at desktop, 297×46 px on a phone |

## Phase 8 — Data cleanup (`fd48534`)

| # | Change | Files | Verified |
|---|---|---|---|
| 32 | `tools/delete_qa_test_data.py`:<br>• matches the exact QA fighter names in one account (`--email`);<br>• lists every fighter, fight and file it would delete;<br>• deletes only with `--confirm`;<br>• logs each deletion as a security event.<br><br>`tools/seed_qa_account.py` creates a separate QA account (password from an environment variable) and refuses any account holding real data. | `tools/delete_qa_test_data.py`, `tools/seed_qa_account.py`, `core/db.py` (`delete_fighters`) | Phase 8 tests: the dry run changes nothing; near-miss names, other accounts and unfiled fights are untouched. **Not run on production.** See OPS_TODO. |
| 52 | Pricing shows one roster notice above the cards instead of one per card. | `pricing.html`, `system.css` | Phase 8 tests; browser |

## Phase 9 — Legal (`f052af6`)

| # | Change | Files | Verified |
|---|---|---|---|
| 9 | `WARRIORIQ_LEGAL_IDENTITY` is one JSON value (name, address, registration, VAT, country, support, privacy and copyright emails) read by every legal page. The older single settings still work. `.env.example` uses @warrioriq.eu addresses. | `core/config.py`, `.env.example`, `render.yaml` | Phase 9 tests (in a fresh interpreter); `/terms` rendered with the configured entity |
| 10 | `/subprocessors` lists Render, Google sign-in and Stripe (when billing is live), each with purpose, location and data processed, plus WarriorIQ's own analysis computer. `render.yaml` names Render (Frankfurt). Policy version is now 2026-10-19. | `core/legal.py`, `core/config.py`, `render.yaml` | Phase 9 tests; rendered page lists Render |

## Phase 10 — Homepage and pricing (`5557f84`)

| # | Change | Files | Verified |
|---|---|---|---|
| 51 | The allowance period is set per plan (`allowance_period`) or globally (`WARRIORIQ_ALLOWANCE_PERIOD`): day (the default, with the current limits), month, or video minutes. Usage gains a nullable `video_seconds` column. The pricing copy follows the period. | `core/payments.py`, `core/db.py`, `app/main.py`, `pricing.html`, `analyze.html`, `sports.html` | Phase 10 tests (minutes and monthly counting; defaults unchanged); existing plan-comparison tests still pass |
| 53 | The A/B boxes are replaced by a demo-clip spot. It plays `WARRIORIQ_DEMO_CLIP` once that file exists; until then a placeholder says the clip is coming. Both link to the sample report. | `index.html`, `app/main.py`, `core/config.py` | Phase 10 tests; browser |
| 54 | `/sample-report`: public, made-up data, labelled as a sample at the top and beside the numbers. It shows only measurements real reports publish: no score and no counts. | `core/sample_report.py`, `sample_report.html`, `app/main.py` | Phase 10 tests; browser at 1280 and 375 px |
| 55 | New homepage sections: what's measured per sport (from `sport_policy`), accuracy figures from the recorded measurements, a pricing summary (from plan config), and two new FAQ entries. | `index.html`, `app/main.py`, `system.css` | Phase 10 tests; browser at 1280 and 375 px |

## Final checks (19 October 2026)

| Check | Result |
|---|---|
| All tests | 2267 passed, 29 skipped, 0 failed |
| Console errors | None, across 14 public pages signed out and 10 pages signed in (Chromium) |
| 8 parallel `/terms` | 0.03–0.10 s total |
| Static page time to first byte | Mean 3–74 ms across `/`, `/pricing`, `/privacy`, `/terms`, `/subprocessors` and `/sample-report` |
| Close fighters / hand-drawn box | Overlapping boxes (IoU 0.79) with one drawn by hand: `warn`, never a block |
| Signed out | `/result/<id>`, `/replay/<id>` and `/progress/<id>` return 303 to `/login?next=…`; `/openapi.json`, `/docs` and `/redoc` return 404 |
| Benchmark ratio (CPU only, no GPU here) | 24 s: 55.8 s, ratio **2.33**; 60 s: 116.8 s, ratio **1.95**; 180 s: 338.1 s, ratio **1.88**. The script exited 1 because every ratio is above 1.0. Nearly all the time is the frame pass on CPU PyTorch. **This does not meet the target here** and says nothing about the GPU promise; see OPS_TODO §1 |

# WarriorIQ upgrade report — 14–15 November 2026

This report answers the owner's brief of 14 November 2026: audit, plan, then implement in tested stages without breaking what works. It covers what was found, what was changed, what was tested, and what is left for the owner.

Everything listed as done was merged through a pull request with the full test suite passing, or is in the final pull request listed at the end. Nothing here was run against production user data.

---

## 1. Technical audit (summary)

The full audit is in the PR history and the session record. Key facts:

| Area | Finding |
|---|---|
| Stack | FastAPI with server-rendered Jinja pages and SQLite, hosted on OuiHeberg cPanel (Passenger). Analysis runs on the owner's GPU PC through `worker.py`, using leases, heartbeats and automatic code updates. |
| Already strong | Login and accounts: PBKDF2 passwords (600k rounds), two-step sign-in, CSRF on every form, strict security headers, rate limits, signed media links. GDPR basics: account export and delete, cookie choices, policy versioning. Analysis: an evidence-gated report and an honest metric catalogue. |
| Critical | 1. The fighters' identities could swap mid-fight (same kit, crossings, small videos). 2. Backups had never been proven. |
| High | 1. Legal documents are unreviewed drafts, with no legal entity yet. 2. Mail is sent from Gmail. 3. There is no published strike-count accuracy. 4. There was no coach access by consent and no gyms. 5. GPU speed is unmeasured. 6. Whether the host has ffprobe is unverified. 7. There are no data processing agreements (DPAs). |
| Medium | 1. `app/main.py` is large (about 11k lines). 2. There was no 30/60/90-day progress view. 3. There was no Ringside Scorer history or export. 4. There were no Muay Thai or MMA pages. 5. `render.yaml` is a leftover from an old host. |

## 2. Problems discovered

The audit list above, plus these, found while doing the work:

1. Deleting a fight's video, whether by hand or by the 30-day sweep, left its browser-playable copy (`<job>_web.mp4`) on disk forever.
2. Pose data from corrections could be exported for training from fights with someone under 18 in them.
3. Profile photos and videos were public files at guessable addresses, including those of private and under-18 profiles. A first fix could be bypassed with an encoded `..` address; this was found by the security review and closed.
4. The request log recorded secret tokens inside addresses: password reset, email verification, guardian approval, email change, share links and invitations.
5. The new crossing still could be requested without a rate limit, and each request decodes video.
6. The backup tool would "back up" an empty database if pointed at the wrong folder.
7. The "Show password" button was 40px wide, under the 44px minimum tap size.
8. The phone-layout audit tool could not sign in, because the login form has a hidden button.

## 3. Problems fixed

Every item in section 2 is fixed. Each fix has a test that fails without it.

The audit's critical item 1 (identity swaps) is addressed as follows:
- The analysis re-runs from the picked frame when its check fails.
- At crossings it asks the person who is who, and re-runs with their answers pinned.
- Small videos are flagged before analysis starts.

It still needs measuring on the owner's labelled fights (section 11).

## 4. Website redesign

The owner chose to keep the approved dark glass design. The homepage was reworked inside it:
- The headline is now "Understand your performance. Improve your game."
- A new "Who it's for" section covers fighters, and coaches and gyms.
- A new "Stays yours" section covers privacy: fights are private by default, videos are deleted after the retention period, training use is opt-in only and never for under-18s, and export and delete are in Settings.

All claims on the page were checked against the code. Two of them uncovered the fixes in section 2.

## 5. New features

| Feature | Where | Notes |
|---|---|---|
| Draw your fighter after crossings | Live analysis page | The analysis keeps going while you draw. The report counts your answer, and "Fix who is who" re-runs the fight with it pinned. |
| 30/60/90-day progress and training consistency | Fight Camp | Built only from fights whose report could tell who was who. Each change needs two fights in the period, otherwise the page says there are not enough. |
| Coaching by consent | `/coaching` (account menu) | The coach invites and the athlete accepts. The coach can read reports (read-only, no video), add notes, and set objectives that appear in the athlete's Fight Camp. Either side can end it. Seats follow the plan. Under-18 accounts cannot accept. |
| Gyms | `/gym` (account menu) | The owner invites coaches and athletes. An athlete who joins lets the gym's coaches read their reports. The owner can remove people and members can leave. Athlete seats follow the owner's plan. |
| Ringside Scorer history and CSV | `/scorer` | Kept on the phone only. Cells are safe in spreadsheets. Each scorecard is marked as unofficial. |
| Admin control room | `/admin` | Alerts that need attention, plus accounts, fights, queue, analysis computer, failures, storage, traffic, safety and tools. Every figure is counted live. |
| Muay Thai and MMA guides | `/muay-thai-fight-analysis`, `/mma-fight-analysis` | Each page says what is measured and what is not observed yet. A test keeps this in step with the sport rules. |
| Small-video warning | Fighter selection | Shown when the video's longer side is under 960px. |
| Daily backups | `tools/backup_runtime.py --keep N` | Integrity-checked. Keeps the newest N and refuses an empty or missing database. |
| Leftover video clean-up | `tools/delete_orphan_web_videos.py` | A dry run by default. With `--confirm` it deletes, logging each deletion. |

## 6. AI analysis improvements

- **Crossings:** each crossing is recorded (`core/crossings.py`). The least certain ones are put to the person, and the answers become identity anchors that re-lock A and B in a re-run (`IdentityManager.apply_person_assignment`). A drawn box is the fighter itself, so it works even when the analysis's own boxes were wrong.
- **Seed check:** a failed check at the picked frame now re-runs the analysis from there, with a named cause in the report.
- **Not added:** high/medium/low confidence badges. The report already labels reliability (estimates, experimental measures, low-confidence identity, withheld numbers), and a new score would have meant inventing numbers.
- **No new strike types:** elbows, clinch, takedowns and submissions still need labelled data and training. The pages say "not observed yet".

## 7. Security improvements

- Profile media is served through an access check, and the old public path is blocked by the resolved file location (not the address text).
- Secret tokens are masked in request logs.
- The crossing still is rate-limited.
- Coaching and gym invitations:
  - only a digest of each token is stored;
  - they expire after 14 days and are single use;
  - every form is CSRF-protected and rate-limited;
  - every access is checked on the server;
  - all of it is recorded in the security log.
- The Strix security review ran on each pull request, with a re-review after fixes. Every finding was fixed, or checked and confirmed as already handled.

## 8. Performance improvements

- The admin storage count is cached for 5 minutes, so the page does not walk the report folders on every visit.
- No heavy queries were added to normal pages. The new coaching and gym queries are indexed (`idx_coach_links_*`, `idx_coach_notes_athlete`, `idx_org_members_*`).

## 9. Database changes

All changes are additive and run automatically at start-up (`core/db.py init_db`). Older code ignores them.

| Change | Purpose |
|---|---|
| Table `coach_links` | Coach ↔ athlete invitations and links |
| Table `coach_notes` | Coaches' notes to athletes |
| Column `coach_assignments.assigned_by_profile_id` (nullable) | Which coach set an objective. Existing rows are null. |
| Tables `organisations`, `org_members` | Gyms and their members |

- **Account deletion:** clears all of the above. The test that lists every table holding account data covers them.
- **Data export:** includes them, without token digests.

## 10. New integrations

None. No new services, packages or paid APIs were added to the website. The Python `playwright` package is only needed to run the phone-layout audit tool on a development machine.

## 11. Outstanding requirements (owner)

| Item | Why | How |
|---|---|---|
| Update the analysis PC | `core/` changed (`crossings.py`, `identity.py`, `analyzer.py`, `db.py`) | Run `update-worker.bat`. The worker also updates itself after a refusal. |
| Daily backup cron job, and one restore test | Backups are unproven | `OPS_TODO.md` section 2 |
| One-time leftover video clean-up | Copies left behind before the fix | `OPS_TODO.md` section 2: dry run first, then `--confirm` |
| ffprobe on the host | The upload type check needs it | In the cPanel terminal, run `which ffprobe ffmpeg` |
| Measure identity on labelled fights | Proves the identity fixes | On the PC: `python tools/identity_benchmark.py` (see `OPS_TODO.md` section 3) |
| Measure GPU speed | The speed promise | `python tools/benchmark_realtime.py --lengths 24 60 180` |
| Legal entity, lawyer review, mailboxes on the domain, data processing agreements | Before payments and before removing the "Draft" banner | `OPS_TODO.md` section 5. The privacy policy now also describes coaching and gyms. |
| Payments (last) | Blocked by the row above | Stripe in test mode first |
| Guardian consent for coaching and gyms | Under-18s cannot join yet | A product and legal decision |

## 12. Testing results

- **Full suite:** 2,552 passed and 29 skipped at the last run before this report (`python -m pytest -q -p no:cacheprovider tests`).
- **New test files:**
  - `test_crossings.py`, `test_who_is_who.py`, `test_small_video_warning.py`
  - `test_video_retention_derivative.py`, `test_delete_orphan_web_videos.py`
  - `test_admin_overview.py`, `test_profile_media_private.py`, `test_progress_periods.py`
  - `test_coaching_links.py`, `test_gyms.py`, `test_ringside_history.py`, `test_seo_guides.py`
  - extended tests in `test_core.py`, `test_product.py` and `test_web.py`
- **Phone layout:** 36 pages at 390px (18 signed out, 18 signed in, including the new pages). No overflow, no tap target under 44px, no text under 12px.
- **End to end on a local site with a real sparring clip:**
  1. Upload and pick the fighters.
  2. The live crossing card appears; draw a box around your fighter.
  3. The report shows "Fix who is who"; the re-run applies the anchor.
  4. Coaching: invite, then accept, then notes and objectives.
  5. Gym: create, then invite, then join.
  6. Scorer: score a fight, then download the CSV, then check the history.
- **Not tested here:**
  - the GPU worker on real hardware;
  - email delivery (no mail server in the test environment; the copy-link fallback was tested);
  - live payments.

## 13. Deployment instructions

1. Merge the pull request on GitHub.
2. In cPanel, go to **Git Version Control** and press **Update from Remote**, then **Deploy HEAD Commit**.
3. In cPanel, go to **Setup Python App** and press **Restart**. The stylesheets changed.
4. On the analysis PC, run `update-worker.bat`.
5. Check the deploy:
   - `https://warrioriq.eu/health` shows the new commit;
   - `/admin` → Analysis shows the computer as "Connected";
   - an old profile photo address (`/static/profile/...`) shows "not found".

## 14. Rollback instructions

1. On GitHub, open the merged pull request and press **Revert**, then merge the revert pull request.
2. In cPanel: **Update from Remote**, then **Deploy HEAD Commit**, then **Restart**.
3. On the PC: `update-worker.bat`.

The database changes are additive, so the old code runs on the new database without changes. The new tables stay unused, and nothing needs restoring.

If something is badly wrong with data, restore the latest copy from the backup folder made by `tools/backup_runtime.py`. Stop the app first, replace `warrioriq.sqlite3`, then start it.

## 15. Ongoing infrastructure costs

This work adds no new recurring cost: no new services, servers or paid APIs. The running costs stay what they were:

- the OuiHeberg hosting plan and the Namecheap domain;
- electricity and upkeep for the owner's GPU PC;
- optionally, a mail provider on the domain when the mailboxes are set up.

Figures depend on the owner's own contracts and are not estimated here.

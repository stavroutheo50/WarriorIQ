# OPS TODO — after the audit of 14 October 2026

These items can't be finished in code. Each needs the owner, a deploy,
money, hardware or a lawyer. Nothing here has been done to any live service.

**Where WarriorIQ runs (owner, 2026-11-12): OuiHeberg cPanel, not Render.**
Namecheap holds the DNS. The code comes from GitHub through cPanel's Git Version
Control (Update from Remote, then Deploy HEAD Commit; `.cpanel.yml` copies it and
restarts Passenger). Environment variables go in cPanel's **Setup Python App**
panel, which wins over `.env`. `.cpanel.yml` does not install Python packages:
after a change to `requirements-web.txt`, run pip install in that panel. The
analysis runs on the owner's own GPU PC. Older notes below that say "Render"
mean "the web host".

## 0. After the November 2026 upgrade (WARRIORIQ_UPGRADE_REPORT.md)

- [ ] **Update the analysis PC:** run `update-worker.bat`, because `core/` changed.
- [ ] **Try coaching and gyms with more than one athlete:** seats follow the plan, and the free plan has one. In Setup Python App, set `WARRIORIQ_COMPLIMENTARY_PLANS=<your email>:gym` and restart the app.
- [ ] **Email for invitations:** coaching and gym invitations are emailed only when email works (admin page → Tools → "Send test email to me"). Otherwise the link is shown once to copy and send by hand.
  - Gym **coach** invitations need email to work, and the coach's address to be verified. A coach can read every athlete's reports in the gym, so a forwarded link must not be enough. With email off, invite people as athletes only.
- [ ] **Lawyer review:** the privacy policy now describes coaching and gyms. Include those lines in the lawyer's review.
- [ ] **Decide on under-18s:** they cannot accept a coach or join a gym as an athlete. Allowing it needs guardian consent for coaching, which is a product and legal decision.

## 1. Speed: deploy and measure on the GPU (#1, #23)

- [ ] **Update your GPU worker PC:** run `deploy/update-worker.ps1` (or `update-worker.bat`) on it. It brings the worker to the code on `main` and restarts it. The pose engine for your GPU is built and cached the first time it runs.
- [ ] **Deploy the website and the worker from the same commit.** The release string (`deploy/release.py`, `/health` → `release`) must match. If it doesn't, analyses pause with "Analysis is paused for an update" and the admin page shows a flag.
  - Since 2026-10-19 the worker fixes this itself: when the website refuses it, it downloads the website's analysis code (`/api/worker/analysis-code`), installs it (the old code goes to `core_previous`) and restarts. Run `update-worker.bat` once to get this; after that, website deploys no longer pause analyses. `WARRIORIQ_WORKER_AUTO_UPDATE=0` in the PC's `.env` turns it off.
- [ ] **Measure the speed promise on your GPU PC:** `python tools/benchmark_realtime.py --lengths 24 60 180`.
  - Each ratio must be 1.0 or less; the script exits 1 otherwise.
  - The only measurement so far was on a CPU-only build machine (see AUDIT_FIXES.md, Final checks). It says nothing about GPU speed.

## 2. Hosting (#3, #4, #25)

- [ ] **Hosting limits:** on cPanel, open Resource Usage after some real traffic and look for CPU, memory or entry-process faults (they show up as 503/508, not timeouts). `WEB_CONCURRENCY` does not apply under Passenger.
- [ ] **ffprobe on the host:** in cPanel's Terminal run `which ffprobe ffmpeg`. Without either, the upload type check can't run and lets the file through.
- [ ] **Backups:** confirm the cPanel backup (JetBackup or Backup Wizard) covers the data folder with the SQLite database and the reports, then restore one copy somewhere to prove it works.
- [ ] **Daily database copy (cPanel, Cron Jobs):** add one job, once a day at a quiet hour, with this command.
  - Take the first part from Setup Python App: the "Enter to the virtual environment" line it shows for the app, without the `cd` at the end.
  - The command: `source <that activate path> && cd /home/dchoodxm/warrioriq && python tools/backup_runtime.py /home/dchoodxm/warrioriq_backups --keep 14`
  - A cron job does not get the variables from Setup Python App. If `WARRIORIQ_DATA_DIR` (or `WARRIORIQ_DB_PATH`) is set there, put the same setting before `python`, e.g. `WARRIORIQ_DATA_DIR=<same value> python tools/...`. Without it the tool stops with "No WarriorIQ database at ..." rather than saving an empty copy.
  - It copies the database with SQLite's online backup (safe while the site runs), checks the copy's integrity, and keeps the newest 14 copies. It never touches anything else in that folder, and videos are not included.
  - `/home/dchoodxm/warrioriq_backups` sits outside `public_html`, so it is never served on the web.
  - Restore drill: download one copy, open it with any SQLite viewer, and check that the `accounts` and `fights` tables have rows. Write down the date you did it.
- [ ] **Leftover video copies (one time, after deploying PR #230):** before 2026-11-15, deleting a video left its browser copy (`<job>_web.mp4`) on disk. Clear them with `tools/delete_orphan_web_videos.py`.
  - Run it the same way as the backup job (a cPanel Cron Job, or Setup Python App's run-script box), with the same `WARRIORIQ_DATA_DIR` the website uses.
  - First without `--confirm`: it only lists the files and their size. Check the list (every name ends in `_web.mp4`), then run again with `--confirm`. Remove the cron job afterwards if you used one.
  - Make a database backup first. Each deletion is logged in security_events.
- [ ] **Optional media CDN (Cloudflare R2 or any pull CDN):**
  - Set `WARRIORIQ_MEDIA_CDN_BASE_URL` and a random `WARRIORIQ_MEDIA_SIGNING_KEY` (keep the key out of the repository).
  - Then check that replay start time drops.

## 3. Identity accuracy (#2)

- [ ] **Re-measure on real fights:** re-run the identity check on the real fights that failed before (the audit says about half did) and compare the pass rate.
  - The new re-entry, camera-motion and partial-coverage logic is unit-tested on synthetic tracks only.
- [ ] **Check the worker image** has the ReID and SAM2 weights it needs. If they're missing, re-entry falls back to kit colour.

## 4. QA data (#32)

- [ ] Back up the production database.
- [ ] **Dry run:** `python tools/delete_qa_test_data.py --email <owner email>` lists exactly what would go.
- [ ] Review the list, then run the same command with `--confirm`.
- [ ] **Separate QA account:** `WARRIORIQ_QA_PASSWORD=… python tools/seed_qa_account.py --email <qa address>`. Run QA there from now on.

## 5. Legal and company (#9, #10)

- [ ] **Register the legal entity.**
- [ ] **Set `WARRIORIQ_LEGAL_IDENTITY` in cPanel's Setup Python App** (Environment variables) as one JSON value:
  `{"name": "...", "address": "...", "registration": "...", "vat": "...", "country": "...", "support_email": "support@warrioriq.eu", "privacy_email": "privacy@warrioriq.eu", "dmca_email": "copyright@warrioriq.eu"}`
- [ ] **Set up the mailboxes:** support@warrioriq.eu, privacy@warrioriq.eu and copyright@warrioriq.eu, then stop using the Gmail address.
  - `.env.example` still shows Gmail as the SMTP sender. Move outgoing mail to a provider on the domain and name it in `WARRIORIQ_EMAIL_PROVIDER_NAME`.
- [ ] **Lawyer review** of every legal document.
  - Only after the whole set is reviewed, set `WARRIORIQ_LEGAL_DRAFT=0` to remove the "Draft — not yet reviewed by a lawyer" banner.
  - Don't clear it after a partial review (see the note in `core/config.py`).
- [ ] **Policy version:** if `WARRIORIQ_POLICY_VERSION` is set in cPanel's Setup Python App, update it to `2026-10-19` or remove it so the new default applies. The cookie banner asks visitors again after a policy change; that is intended.
- [ ] **Analysis computer location:** fights are analysed on your own GPU PC, so /subprocessors lists it as WarriorIQ's own "Analysis computer", not a third party. Set `WARRIORIQ_ANALYSIS_PROVIDER_LOCATION` (e.g. the country it is in) so the page states where it is.
- [ ] **Old environment value:** if `WARRIORIQ_ANALYSIS_PROVIDER_NAME` is set in cPanel's Setup Python App (an earlier `render.yaml` set it), delete it, or /subprocessors will keep naming a third-party analysis provider.
- [ ] **Data processing agreement** with OuiHeberg now (and Namecheap if it handles any personal data). With Stripe before billing opens, when Stripe appears on /subprocessors automatically.

## 6. Homepage and pricing (#51, #53, #54, #55)

- [ ] **Allowance period (#51):** decide whether plans stay per day.
  - For per-month: add `monthly_limit` to each plan and set `WARRIORIQ_ALLOWANCE_PERIOD=month`.
  - For video minutes: add `monthly_video_minutes` and set `WARRIORIQ_ALLOWANCE_PERIOD=video_minutes`.
  - A plan without its number keeps counting per day.
- [ ] **Accuracy section (#55):** the figures come from measurements recorded in `core/report.py` and `core/sport_policy.py`. Update the homepage when `tools/benchmark_labelled_fight.py` or the Accuracy Lab results change.
- [ ] **Taekwondo clip licence (2026-11-13):** `app/static/home/taekwondo.mp4` was supplied by the owner from a file named "Video Project 1". Its source is not the US Army Commons video it was thought to be (that one shows a male soldier in Korea; this shows two women sparring in a French training hall). Find where it came from and confirm it can be used commercially (Pexels, Pixabay, CC BY / CC BY-SA / public domain, or written permission), then add the credit to the homepage and sports page. Replace it if that cannot be confirmed.
- [ ] **Guard (#5):** label a set of guard clips by hand. Remove `experimental=True` from the guard metric only once it has been checked against them.

## 7. Account safety (2026-11-12)

- [ ] **Install the new package, then set the key.** In cPanel's Setup Python App: run pip install for `requirements-web.txt` (it now includes `cryptography`; `.cpanel.yml` does not install packages), then add `WARRIORIQ_TWO_FACTOR_KEY` as an environment variable (a long random passphrase; keep a copy in a password manager) and restart the app. The key encrypts every account's authenticator secret in the database. Without the package the key is ignored, secrets are stored as plain text and a warning is logged. If the key is later lost or changed, authenticator codes stop working until the person uses a recovery code and sets two-step sign-in up again.
- [ ] **No shell on the hosting plan (2026-11-12): cPanel's Terminal says "Shell access is not enabled".** Install the package from the Python App page instead: under **Configuration files** add `requirements-web.txt` and click **Run Pip Install** (it only installs what is missing; the deployed file lists `cryptography>=42`). Check the result on the admin page, **Two-step sign-in storage**: it says *Encrypted*, or says exactly why not (key missing, package missing) and counts how many accounts store their secret encrypted or in plain text. If shell access is ever wanted, ask OuiHeberg support to enable it.
- [ ] **Try it once on the live site** with your own account: turn two-step sign-in on, sign out, sign in with a code, then with a recovery code, and check the devices list shows this browser.


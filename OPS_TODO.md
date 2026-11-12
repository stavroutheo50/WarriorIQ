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
- [ ] **Guard (#5):** label a set of guard clips by hand. Remove `experimental=True` from the guard metric only once it has been checked against them.

## 7. Account safety (2026-11-12)

- [ ] **Install the new package, then set the key.** In cPanel's Setup Python App: run pip install for `requirements-web.txt` (it now includes `cryptography`; `.cpanel.yml` does not install packages), then add `WARRIORIQ_TWO_FACTOR_KEY` as an environment variable (a long random passphrase; keep a copy in a password manager) and restart the app. The key encrypts every account's authenticator secret in the database. Without the package the key is ignored, secrets are stored as plain text and a warning is logged. If the key is later lost or changed, authenticator codes stop working until the person uses a recovery code and sets two-step sign-in up again.
- [ ] **Try it once on the live site** with your own account: turn two-step sign-in on, sign out, sign in with a code, then with a recovery code, and check the devices list shows this browser.


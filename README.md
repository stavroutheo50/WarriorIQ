# WarriorIQ — consolidated build

This build consolidates the WarriorIQ requirements discussed across the project instead of continuing the old patch-on-patch test scripts.

## What is included

- Full fight-video upload.
- Competition or sparring mode, including footage with no referee.
- Fighter A and Fighter B selected separately from a real video frame.
- Automatic person boxes plus manual draw correction.
- Analyze A, B or both.
- WarriorIQ identity manager independent of temporary BoT-SORT IDs.
- Referee/coach exclusion by identity evidence and uncertainty instead of guessing.
- Short-window SAM2.1 emergency identity recovery only when the fast tracker becomes uncertain.
- YOLO26 Pose on NVIDIA CUDA, with optional TensorRT engine.
- Entire scheduled fight / selected rounds; K-1, Low Kick, Full Contact, Point Fighting, Light Contact and Kick Light.
- Accurate frame-based progress, current speed and ETA.
- Reproducible fixed FPS/resolution by default; adaptive speed mode is an explicit environment opt-in.
- Multi-frame action state machine: no single-frame “hand moved fast = punch” events.
- Optional trained WarriorIQ pose-transformer temporal model checkpoint, with fight-group validation and backwards-compatible GRU checkpoint loading.
- Jab/cross/hooks/uppercuts, round/front/low/body/head kicks and knees.
- Attempted actions plus clean/likely landed/blocked/checked/missed/uncertain outcomes.
- Head/body/leg target estimate.
- Defense evidence: block/check plus temporal parry/slip/evade when supported.
- Combinations and counters.
- Pose coverage, activity evidence, footwork, pressure, ring-center control, guard and balance metrics when coverage is sufficient.
- Evidence-based strongest weapon and vulnerability areas.
- Ruleset-aware estimated scorecards.
- Deep coaching: strengths, improvements, drills and training plan tied to evidence.
- Original fight replay with timeline jump buttons.
- Skeleton overlay and skeleton-only replay; only Fighter A/B are drawn.
- Saved local athlete profile/photo/notes, fight history, comparison and coach portal.
- Private adult athlete accounts with salted password hashes, expiring local sessions and one-time password-reset tokens.
- Optional enforced email verification with 24-hour one-time links and a rate-limited resend flow.
- Durable analysis leases plus a separate claimable GPU worker process; local PyCharm keeps an automatic in-process worker.
- Operational `/ready` checks for the database, storage reserve, queue and worker, while `/health` remains a minimal uptime probe.
- Configurable upload malware scanning, duration/resolution limits, generated storage names and a minimum free-space reserve.
- Temporary guest analyses that are excluded from history and removed after two hours.
- Athlete progress trends that preserve missing/unavailable measurements.
- Coach assignments with active/completed status.
- Seven-day, revocable, video-free report-summary links.
- Automatic preflight checks for resolution, frame rate, lighting and motion clarity.
- Credit-aware Stripe Checkout and signed webhook foundations, disabled until configured.
- Internal JSON/HTML analysis artifacts; raw exports are not exposed in the public interface.
- No fake/demo statistics. Unsupported/low-evidence values remain unavailable/uncertain.
- Privacy settings for account export, separate video/analysis/account deletion, consent choices and session revocation.
- Standard Stripe-ready architecture, disabled by default and requiring a lawful configured account.



## First professional-workspace setup

Open **Create account** in the local website. The first local account safely claims the existing athlete profile and its saved fight library, so previous analyses are not lost. Later local accounts receive separate profiles and cannot access another profile's library, comparisons, assignments, media or reports.

Without an account, analysis still works as a temporary guest job. Guest files are retained only long enough to render the result and are automatically removed after two hours.

## Payments

Pricing is a founding-beta preview while `WARRIORIQ_PAYMENTS` is disabled. The entitlement contract is Starter (1 analysis/day, Compact report), Athlete (3/day, Expanded report), Athlete Pro (10/day, Complete report), Coach (30/month, Complete report), and Gym (unlimited, Complete report). Daily and monthly periods reset on UTC boundaries. Reservations are atomic and failed analyses return their reserved use.

Before enabling payments, configure the matching Stripe price IDs—including `STRIPE_PRICE_GYM`—plus `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET`. Checkout metadata links a verified Stripe session to the signed-in local account. A public launch still needs complete subscription-renewal, cancellation, tax, invoice and legal testing in the intended jurisdiction.

## Legal and public-launch gate

WarriorIQ now includes a Legal Center plus Privacy, Terms, Cookie, Acceptable Use, Refund/Cancellation, EULA, Copyright/DMCA, Accessibility, AI Transparency, Security and Service Provider pages. It also serves `robots.txt`, `sitemap.xml`, private-page `noindex` controls, social metadata and custom error pages.

Copy the real values from `.env.example` into your deployment environment. The code intentionally does not invent a company name, address, registration number, jurisdiction or monitored contact. Paid checkout returns a safe 503 response until every launch-critical operator field is configured with a public HTTPS base URL.

Signup is currently 18+ and separately records Terms, Privacy, age and optional marketing choices. The database includes a future guardian-approval status, but junior accounts are not enabled. Every fight upload records footage rights, permissions for people shown and either no-minor or appropriate guardian-permission confirmation; optional OpenAI frame processing has a separate record. Corrections are private by default; model-training sequence export happens only after a signed-in user enables it on Athlete Profile. Account export and deletion are password protected.

Original signed-in videos are scheduled for deletion after `WARRIORIQ_VIDEO_RETENTION_DAYS` (30 by default), while reports can be retained separately. Guests expire after two hours, and abandoned processing artifacts are cleaned after `WARRIORIQ_FAILED_UPLOAD_RETENTION_HOURS`. See `COMPLIANCE_OPERATIONS.md` for the provider inventory and remaining operational launch blockers.

Email verification is implemented but deliberately disabled until a real SMTP provider and monitored sender are configured. Set `WARRIORIQ_EMAIL_PROVIDER=smtp`, the SMTP variables, and then `WARRIORIQ_REQUIRE_EMAIL_VERIFICATION=true`. Enabling the requirement without working delivery would prevent new users from entering their workspace.

### Optional social sign-in

Google, Apple, Facebook and Microsoft sign-in use the authorization-code flow and stay hidden until `WARRIORIQ_OAUTH_STATE_SECRET` plus that provider's client ID and secret are configured. Generate a long, private state secret and register only these exact production callbacks:

```text
https://warrioriq.eu/auth/google/callback
https://warrioriq.eu/auth/apple/callback
https://warrioriq.eu/auth/facebook/callback
https://warrioriq.eu/auth/microsoft/callback
```

Use the provider's web-application configuration, keep secrets server-side, and add separate localhost callbacks only for controlled development. Apple requires a Services ID associated with the website and a signed client-secret JWT; rotate that JWT before its configured expiry. WarriorIQ stores the provider name and stable subject identifier but not provider access tokens. It deliberately does not auto-link an unconnected provider to an existing password account solely because the email matches.

Read `PRODUCTION_LAUNCH_CHECKLIST.md` before any public deployment. Passing the code-level gate does not replace legal, security, accessibility, payment, AI validation or operational sign-off.

## Render deployment

The repository includes `render.yaml`. Render should build with `pip install -r requirements.txt`, start with `python run.py`, and probe `/health`. `run.py` reads Render's `PORT`, binds to `0.0.0.0`, trusts proxy headers through Uvicorn, and does not open a desktop browser in production.

Set `WARRIORIQ_PUBLIC_BASE_URL` to the final public HTTPS origin, for example `https://warrioriq.onrender.com`. Add the verified operator and legal-contact values listed in `.env.example` before paid public launch. Render supplies `RENDER` and `PORT`; do not create replacements for them. The web interface now starts without importing Torch, Ultralytics, or SAM2. Fighter selection remains available through manual box drawing when optional candidate detection is disabled or unavailable.

The default Render profile disables selection-page YOLO detection and SAM2 recovery to protect a small CPU instance from avoidable startup and memory pressure. A capable analysis worker can opt in with `WARRIORIQ_SELECTION_DETECTION=true`, `WARRIORIQ_SAM_RECOVERY=true`, and `WARRIORIQ_SAM_CONTINUOUS=true`. These are quality/performance controls, not requirements for the website to load.

Render's free filesystem is ephemeral. Without a persistent disk, accounts, uploaded videos, analyses, and the SQLite database can disappear after a restart or redeploy. On a paid service with a disk mounted at `/var/data`, set `WARRIORIQ_DATA_DIR=/var/data/warrioriq`. Do not set that path unless the disk is actually mounted and writable.

### OuiHeberg cPanel with Namecheap DNS

`warrioriq.eu` currently uses Namecheap nameservers while the purchased application hosting is OuiHeberg cPanel. The repository is checked out at `/home/dchoodxm/warrioriq_repo`; `.cpanel.yml` deploys tracked application files into the Python application's stable root at `/home/dchoodxm/warrioriq` without copying Git metadata, local secrets or generated uploads. In CloudLinux **Setup Python App**, use `warrioriq_wsgi.py` as the application startup file and `application` as the entry point; `passenger_wsgi.py` remains available for hosts that load a Passenger WSGI file directly. Never select `passenger_wsgi.py` as its own CloudLinux startup module because the generated Passenger wrapper would recursively import itself. Install `requirements-web.txt` in the shared web host; it intentionally excludes the Torch/SAM/Ultralytics GPU stack. If the panel's pip button does not run, execute `install_web_dependencies.py` through its Python-script control. Install the full `requirements.txt` only on the analysis worker. Do not upload a local `.env`; create production variables through **Setup Python App** and set at minimum `WARRIORIQ_PUBLIC_BASE_URL=https://warrioriq.eu`. Keep payments disabled until every legal/operator field and Stripe webhook is configured and verified.

After each GitHub update, use cPanel Git Version Control to **Update from Remote** and then **Deploy HEAD Commit**. Restart the Python application, run `/health`, and enable AutoSSL for both `warrioriq.eu` and `www.warrioriq.eu`. The domain must use the DNS nameservers supplied by OuiHeberg before certificates and traffic can reach this application. The public site is not ready while either hostname has a certificate error. Never place GitHub credentials or production secrets in the repository.

This repository also remains compatible with hosts that provide `PORT` or `SERVER_PORT`. `run.py` binds to `0.0.0.0` in those environments, while Passenger uses the separate `passenger_wsgi.py` WSGI adapter.

Shared hosting is suitable for serving and testing the web product, but the complete Torch/Ultralytics/SAM2 fight-analysis workload may exceed its memory, process-time or CPU limits. Confirm those limits with the hosting plan before enabling public uploads. If the analysis worker cannot finish a two-minute video reliably, keep the domain and web layer here and run analysis on a dedicated GPU worker instead of silently reducing analysis quality.

### Durable GPU worker

Local development uses `WARRIORIQ_WORKER_MODE=inprocess`. A production runtime with a shared private data mount can set `WARRIORIQ_WORKER_MODE=external` on the web process and run:

```text
python worker.py
```

Queued jobs remain persisted across web restarts. A worker claims one job at a time, renews a lease during progress, and leaves an interrupted job restartable if the lease expires. Because that queue is durable, `external` and `remote` modes accept a fight while the analysis machine is switched off and start it automatically when the worker next connects; the uploader is told the fight is waiting rather than being refused. Set `WARRIORIQ_ACCEPT_DEFERRED_ANALYSIS=false` to refuse uploads whenever no worker is online. A server that could never process the fight at all — missing vision dependencies, no worker token, an invalid mode — still refuses immediately instead of queueing work forever. The web and worker must point `WARRIORIQ_DATA_DIR` at the same protected runtime storage. Do not claim that a remote GPU is connected merely because external mode is enabled; `/ready` returns 503 until a worker heartbeat is present.

### Scale-to-zero GPU (analysis without a machine left running)

A fight needs a GPU for the few minutes it is actually being analysed, not for the rest of the day. Serverless GPU platforms (Modal, RunPod Serverless, Beam and similar) bill per second and scale to zero, so the cost of an idle service is nothing.

Set `WARRIORIQ_WORKER_WAKE_URL` on the web server to the endpoint that starts your worker container. When a fight is queued, the web process pings that URL with the same `WARRIORIQ_WORKER_TOKEN` bearer token, the container starts, runs `python worker.py --once` against `WARRIORIQ_WORKER_REMOTE_URL` until the queue is empty, and exits. Nothing polls a running GPU, so nothing bills idle time.

The wake call is best effort and runs off the request thread: the fight is already queued durably, so a slow or failed wake never fails the upload and the worker still claims the fight on its next start. Leave the variable empty when a machine polls continuously, such as a PC running `worker.py`.

Bake the model weights into the container image or a persistent volume. Downloading YOLO and SAM2 on every cold start would dominate both the wait and the bill. Confirm current per-second pricing with the platform before relying on any cost estimate.

When the GPU is on a different machine, set `WARRIORIQ_WORKER_MODE=remote` and a high-entropy `WARRIORIQ_WORKER_TOKEN` on the web server. On the GPU machine, install the full requirements and model assets, then set `WARRIORIQ_WORKER_REMOTE_URL=https://warrioriq.eu` plus the same token and run `python worker.py`. The worker downloads only its claimed video over authenticated HTTPS, streams genuine progress, and uploads a bounded report/tracking archive. Keep the token server-side, rotate it if exposed, and confirm `/ready` is healthy before accepting public analyses.

`/health` answers only whether the web process is alive. Use `/ready` for deployment readiness. Before backups, choose a protected destination outside the public web root and run `python tools/backup_runtime.py <destination>`. The command uses SQLite's online backup API, runs an integrity check and writes a SHA-256 manifest; it intentionally excludes original videos.

## Performance requirement

The hard product target is **<= 1.0x wall time**: a 120-second entered fight segment should finish analysis in <=120 seconds. WarriorIQ records whether the test passes; it does not fake a pass.

The responsive website can be used from a phone or computer. Analysis runs on the server computer: NVIDIA CUDA/TensorRT is fastest, while the pose pipeline can fall back to CPU at substantially lower speed. For additional NVIDIA speed, export the pose model to TensorRT:

```powershell
.\.venv\Scripts\python.exe tools\export_pose_tensorrt.py
```

## What code alone cannot truthfully guarantee

A custom kickboxing classifier cannot become “best in the market” merely by writing a neural-network class. It needs a large, correctly labeled kickboxing dataset, held-out complete-fight validation, and a separate untouched test set. Until a checkpoint passes those release gates, WarriorIQ treats automatic detections as private review candidates and exposes only human-confirmed actions as evidence.

The acceptance benchmark remains: complete unseen fights end-to-end, no manual correction after initial A/B selection, referee/no-referee footage, stable identities, attacks/outcomes, ruleset scoring, statistics and useful evidence-linked reports — while meeting the real-time budget.

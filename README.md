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

## Install on the WarriorIQ Windows PC

Open this folder in PyCharm, open Terminal, then:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_windows.ps1
```

Verify the project before the first run:

```powershell
.\.venv\Scripts\python.exe tools\verify_project.py
```

Only continue if the project check passes. Then run:

```powershell
.\.venv\Scripts\python.exe run.py
```

The browser should open `http://127.0.0.1:8000` automatically.

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

Read `PRODUCTION_LAUNCH_CHECKLIST.md` before any public deployment. Passing the code-level gate does not replace legal, security, accessibility, payment, AI validation or operational sign-off.

## Performance requirement

The hard product target is **<= 1.0x wall time**: a 120-second entered fight segment should finish analysis in <=120 seconds. WarriorIQ records whether the test passes; it does not fake a pass.

The responsive website can be used from a phone or computer. Analysis runs on the server computer: NVIDIA CUDA/TensorRT is fastest, while the pose pipeline can fall back to CPU at substantially lower speed. For additional NVIDIA speed, export the pose model to TensorRT:

```powershell
.\.venv\Scripts\python.exe tools\export_pose_tensorrt.py
```

## What code alone cannot truthfully guarantee

A custom kickboxing classifier cannot become “best in the market” merely by writing a neural-network class. It needs a large, correctly labeled kickboxing dataset, held-out complete-fight validation, and a separate untouched test set. Until a checkpoint passes those release gates, WarriorIQ treats automatic detections as private review candidates and exposes only human-confirmed actions as evidence.

The acceptance benchmark remains: complete unseen fights end-to-end, no manual correction after initial A/B selection, referee/no-referee footage, stable identities, attacks/outcomes, ruleset scoring, statistics and useful evidence-linked reports — while meeting the real-time budget.

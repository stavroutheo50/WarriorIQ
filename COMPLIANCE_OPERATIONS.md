# WarriorIQ compliance operations (implementation record)

Status: internal draft for technical, security, privacy and legal review. This is not approved legal advice or a completed launch assessment.

## Product position

WarriorIQ is an AI-assisted combat-sports training and performance analytics platform. It is not a medical diagnostic system, official judging authority, guaranteed scoring system, biometric identification service or gambling/prediction service.

## Service/provider inventory

Complete this table with the contracted legal entity, region, agreement/DPA, retention, transfer mechanism and production owner before any provider receives user data.

| Function | Current implementation | Data that may be received | Launch requirement |
|---|---|---|---|
| Hosting | Local FastAPI process; public host not selected | HTTP requests, account and application data, depending on deployment | Select provider/region and document controls |
| Database | Local SQLite; production database not selected | Account, consent, fight metadata, billing status, audit records | Managed encrypted database, backup/restore and access policy |
| Authentication | WarriorIQ PBKDF2 password hashes and expiring digest-backed sessions | Email, password hash, sessions | Security review; transactional email for reset links |
| Video storage | Private server filesystem with owner-scoped routes | Original videos, selection frames, generated overlays | Private object storage, deny-by-default policy and signed URLs if used |
| AI | Local Ultralytics/PyTorch/SAM; OpenAI recovery only after separate per-upload opt-in | Local models receive video; OpenAI may receive selected frames when enabled | Provider agreement, region/retention review, no training without opt-in |
| Email | Not configured | Email and message content once configured | Configure `WARRIORIQ_EMAIL_PROVIDER` and delivery worker |
| Analytics | None installed | None | Do not add before consent gating and policy update |
| Payments | Stripe integration disabled by default | Email, plan, checkout and payment metadata; Stripe processes card data | Configure lawful Stripe account, prices, signed webhook and tax settings |

## Privacy and storage controls

- Fight videos, reports and profiles are private by default. There is no public video feed or automatic broadcast/stream downloader.
- `/media/{job_id}` and other fight routes require the current guest or signed-in owner. IDs alone do not grant access.
- Original signed-in videos are scheduled for deletion after `WARRIORIQ_VIDEO_RETENTION_DAYS` (default 30); reports may be retained separately. Guests expire after two hours.
- Account deletion removes account/profile/video/report data and de-identifies local security events. Any production exception for legally required billing records needs a documented schedule.
- Model-training permission is a separate, default-off profile choice. It covers deliberate correction feature sequences, not automatic reuse of raw customer videos.
- Fighter tracking distinguishes the two user-selected subjects only within one upload. Do not add face identification, real-world identity lookup or persistent biometric profiles.

## Security and incident preparation

- Deploy only behind HTTPS. The app redirects to the configured HTTPS base URL and emits HSTS for HTTPS responses.
- Keep Stripe, AI, email, database and storage secrets server-side. Never put service-role or secret keys in templates or frontend bundles.
- Use a shared edge/application rate limiter in production; the built-in limiter is a single-process safety layer.
- Replace local filesystem video storage with a private bucket policy and short-lived signed delivery if the production architecture uses object storage.
- Monitor `security_events`; explicitly configure administrators through `WARRIORIQ_ADMIN_EMAILS`. Admin access and account status changes are recorded.
- Maintain an incident runbook: contain access, revoke sessions/links/keys, identify affected accounts/resources, preserve required evidence, notify advisers/regulators/users within applicable deadlines, and document recovery.
- Test restore, session revocation, account suspension, media isolation, malicious upload handling and copyright removal before launch.

## Operational launch blockers

Code scaffolding does not complete these external obligations: final policies and EU withdrawal wording; business identity/tax/VAT setup; payment and email configuration; malware scanning; production object-storage policies; processor agreements; DPIA/legitimate-interest analysis where required; child-safety programme before junior accounts; accessibility audit; model validation; breach/complaint staffing; backup retention; insurance; and jurisdiction-specific consumer review.

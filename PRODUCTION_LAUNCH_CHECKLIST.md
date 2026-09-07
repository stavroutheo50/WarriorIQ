# WarriorIQ public-launch checklist

This checklist is an engineering and operations control, not a substitute for legal advice. Do not mark the public product ready merely because the policy pages render.

## Hard launch gate

- [ ] Form the real operating entity and verify its legal name, registration, address, VAT/tax status and governing jurisdiction.
- [ ] Configure every `WARRIORIQ_*` identity field in `.env.example`; use a public HTTPS URL and monitored support, privacy and copyright inboxes.
- [ ] Have qualified counsel review the Terms, Privacy Policy, EULA, Acceptable Use, Refund/Cancellation, copyright/DMCA, minors and consumer checkout for every launch jurisdiction.
- [ ] Register and maintain a designated agent with the U.S. Copyright Office if relying on the DMCA safe-harbour process.
- [ ] Decide whether minors are prohibited or supported. Supporting minors requires age assurance, verified guardian authority, safeguarding and a separately reviewed privacy design.
- [ ] Keep paid checkout disabled until the `/legal` page reports that the code-level identity gate is configured and the operational checks below are signed off.

## Privacy and AI governance

- [ ] Build a record of processing activities covering accounts, footage, profiles, tracking, corrections, billing, support, sharing, logs and backups.
- [ ] Document purpose, legal basis, data categories, recipients, retention and deletion for every processing activity.
- [ ] Complete and approve a DPIA before large-scale or otherwise high-risk video/AI processing where required.
- [ ] Sign processor/data-protection agreements with hosting, storage, email, monitoring, OpenAI, Stripe and any future provider; document regions and international-transfer safeguards.
- [ ] Publish the actual provider list before any provider receives production personal data.
- [ ] Define production retention for accounts, payments, acceptance evidence, logs, deleted items and backups; test deletion through backups.
- [ ] Keep external identity recovery off by default and preserve a per-upload acceptance record when it is enabled.
- [ ] Keep model-training export separate and off by default; document how existing exported sequences are handled after withdrawal.
- [ ] Maintain an AI system inventory, intended purpose, model/data provenance, evaluation sets, known limitations, change log and incident/escalation process.
- [ ] Validate fighter identity continuity, technique, side, target, outcome, timing and ruleset legality with representative labelled fight data. Never convert tracking coverage into a claim of action accuracy.
- [ ] Preserve clear automated, preliminary and human-reviewed labels. Do not market WarriorIQ as an official judge or guaranteed safety tool.

## Payments and consumers

- [ ] Configure Stripe in live mode with least-privilege keys stored in a managed secret store.
- [ ] Verify webhook signatures, idempotency, retries and reconciliation in the live account.
- [ ] Show the exact recurring price, taxes, billing period, included analyses, renewal and cancellation terms before checkout.
- [ ] Ensure the final checkout control unambiguously communicates an obligation to pay.
- [ ] Implement a customer billing portal or monitored cancellation route and test cancellation, plan changes, duplicate charges, refunds and failed payments.
- [ ] Implement jurisdiction-specific withdrawal/cooling-off handling. If service begins immediately, obtain any required separate express request and acknowledgement; do not pre-check it.
- [ ] Publish trader/contact information and complaint-handling details required in each market.

## Security and resilience

- [ ] Run production behind HTTPS only; enable HSTS after confirming every subdomain supports HTTPS.
- [ ] Store secrets outside source control; rotate them; separate development, staging and production.
- [x] Add rate limiting for signup, login, uploads, analysis start, sharing, export and support endpoints.
      *(Code side done 2026-09-07. Limits count against the visitor, not the proxy: keying on `request.client.host` put every
      visitor behind Apache in one bucket. Still single-process, so an edge or shared limiter is the remaining piece.)*
- [x] Add CSRF protection for every authenticated state-changing form and API, including login-CSRF protection.
      *(Done 2026-09-07. Double submit with an httponly cookie; a test walks the route table so a new route cannot skip it.
      Exempt: the worker API, the Stripe webhook, and the OAuth callback — each authenticates its own way and has no cookie.)*
- [ ] Add email verification, password reset, session/device management and optional MFA before public accounts.
- [ ] Scan uploads for malware, validate decoded media, isolate media processing, keep generated storage names and enforce byte/duration/resolution limits.
- [ ] Encrypt production data at rest and in transit; restrict staff access by role; maintain auditable access logs without logging raw video or secrets.
- [ ] Create incident response, vulnerability disclosure, breach assessment/notification and law-enforcement request procedures with named owners.
- [ ] Configure monitoring, backups, restore drills, job recovery, dependency scanning, patch SLAs, WAF/DDoS controls and an availability plan.
- [ ] Commission an independent security review and remediate material findings before launch.

## Accessibility and product quality

- [ ] Test keyboard-only use, focus order, screen readers, 200–400% zoom, reduced motion, colour contrast, form errors and mobile orientation against WCAG 2.2 AA.
      *(Partly addressed. Measured in a browser on 2026-09-07 across the public pages: **contrast** passes AA
      everywhere checked (0 failures), every control has an accessible name, every input has a label, no image
      is missing alt text and no heading level is skipped. **Reflow at 320px** — the same as 400% zoom — found
      one real failure on the home page, now fixed. A skip link, reduced-motion rules and focus-visible styles
      were already in place, and fighter selection was made keyboard-operable. **Still untested: an actual
      screen reader, and manual testing by people who use assistive technology.** Automated checks are a floor,
      not this item.)*
- [x] Provide an accessible alternative for canvas fighter selection and skeleton overlays.
      *(Done 2026-09-07. Fighter selection can be completed from a described list of detected people instead of drawing —
      without it the whole product was unusable by keyboard. The skeleton overlay is marked decorative: the replay chapter
      buttons already carry everything it conveys that anybody can act on.)*
- [ ] Caption instructional media and provide transcripts where audio conveys information.
- [ ] Test every CTA, link, form, upload, checkout, replay control, deletion/export flow and error page on supported browsers and devices.
- [ ] Run automated accessibility checks plus manual testing by people who use assistive technology.
- [ ] Publish known accessibility limits and maintain a monitored remediation/feedback workflow.

## Web, SEO and release operations

- [ ] Configure the final base URL, canonical URLs, production `robots.txt`, `sitemap.xml`, page titles/descriptions, favicon and social-share image.
- [ ] Keep private reports, shares, media, account pages, APIs and job pages `noindex` and access-controlled.
- [ ] Add privacy-respecting analytics only after defining the purpose and consent gate; do not load optional trackers before consent.
- [ ] Verify custom 403/404/410/413/429/503 pages, broken-link scanning, form validation, mobile layout and performance budgets.
- [ ] Minify/cache static assets safely, optimise images, set upload/analysis capacity limits and test a realistic two-minute fight under peak load.
- [ ] Create staging, release approval, rollback and database migration procedures. Back up before schema changes and test rollback.
- [ ] Keep a release evidence packet containing test results, accessibility results, security review, model validation, legal sign-off and named go/no-go approval.

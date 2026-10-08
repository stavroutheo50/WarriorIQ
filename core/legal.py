from __future__ import annotations

from core.config import SETTINGS


REQUIRED_LAUNCH_FIELDS = {
    "public_base_url": "Public HTTPS website URL",
    "operator_name": "Legal operator or company name",
    "operator_address": "Legal business address",
    "operator_registration": "Business registration number",
    "governing_country": "Governing country/jurisdiction",
    "support_email": "Customer-support email",
    "privacy_email": "Privacy-rights email",
    "dmca_email": "Copyright/DMCA email",
    "dmca_agent_name": "Designated copyright agent name",
    "email_provider": "Transactional email provider",
}


def launch_readiness() -> dict:
    missing = [label for field, label in REQUIRED_LAUNCH_FIELDS.items() if not getattr(SETTINGS, field, "")]
    if SETTINGS.public_base_url and not SETTINGS.public_base_url.lower().startswith("https://"):
        missing.append("Public URL must use HTTPS")
    return {
        "ready": not missing,
        "missing": missing,
        "policy_version": SETTINGS.policy_version,
        "operator": {
            "name": SETTINGS.operator_name,
            "address": SETTINGS.operator_address,
            "registration": SETTINGS.operator_registration,
            "vat": SETTINGS.operator_vat,
            "country": SETTINGS.governing_country,
            "support_email": SETTINGS.support_email,
            "privacy_email": SETTINGS.privacy_email,
            "dmca_email": SETTINGS.dmca_email,
            "dmca_agent_name": SETTINGS.dmca_agent_name,
        },
    }


LEGAL_DOCUMENTS = {
    "terms": {
        "title": "Terms of Service",
        "description": "The rules for using WarriorIQ's fight-analysis website and athlete workspace.",
        "intro": "These Terms govern the WarriorIQ website and analysis service and take effect when a user accepts the policy version shown below.",
        "sections": [
            ("Who may use WarriorIQ", f"People of any age may create an account. Under {SETTINGS.minimum_account_age}, a parent or guardian must approve the account through a link WarriorIQ emails to them, and the account cannot upload fight footage until they do; by approving, they accept these Terms on the young person's behalf. Fight footage may show other junior athletes only when the uploader has every required right and appropriate parent or guardian permission."),
            ("Your footage and permissions", "You keep ownership of your footage. You give the operator a limited licence to store, decode, process, display and delete it only as needed to provide the service. Before uploading, you must have the rights to the recording and an appropriate legal basis or permission for every identifiable person shown. A checkbox does not replace consent or another legal basis required by local law."),
            ("Analysis limitations", "WarriorIQ produces training and coaching support, not an official judging decision, medical assessment, safety guarantee or substitute for a qualified coach. Automated labels, identities and estimated scorecards can be wrong. Confidence labels and evidence replay must be considered before acting on a result."),
            ("Accounts and security", "Provide accurate account information, protect your password, and tell support about suspected unauthorised access. A parent or legal guardian managing a child's workspace is responsible for its settings, uploads and sharing choices unless the law says otherwise."),
            ("Plans, billing and cancellation", "Paid plans renew for the period shown at checkout until cancelled. The price, taxes, analysis allowance, report depth and renewal terms must be presented before payment. Cancellation stops future renewal and does not erase legal refund or withdrawal rights. The Refund and Cancellation Policy forms part of these Terms."),
            ("Acceptable use", "Do not upload unlawful, stolen, abusive or secretly recorded content; identify or harass people; bypass access controls or allowances; scrape the service; reverse engineer protected components; distribute malware; or use the service to make high-impact decisions about another person. The Acceptable Use Policy gives more detail."),
            ("Suspension and termination", "Access may be restricted when reasonably necessary to protect users, the service, legal rights or security. Where practical and lawful, WarriorIQ explains the reason and gives a route to appeal. Users may delete their account from the Athlete Profile."),
            ("Intellectual property", "WarriorIQ branding, interface and original software are protected by applicable intellectual-property law. These Terms do not transfer them. User footage and lawful feedback remain subject to the rights described in these Terms and the Privacy Policy."),
            ("Disclaimers and liability", "The service is provided with the care required by applicable law, but uninterrupted operation and perfect analysis are not promised. Nothing in these Terms excludes rights or liability that cannot legally be excluded, including mandatory consumer protections."),
            ("Changes and contact", "Material changes require a new policy version and renewed acceptance when appropriate. The configured operator and governing jurisdiction appear in the Legal Notice. Questions can be sent to the configured support address."),
        ],
    },
    "cookies": {
        "title": "Cookie Policy",
        "description": "What WarriorIQ stores in the browser and how cookie preferences work.",
        "intro": "WarriorIQ uses essential browser storage for security and private job access. Optional analytics storage is installed and stays off until you accept it. Advertising storage and ad personalisation are refused at all times, including when you choose Accept All.",
        "sections": [
            ("Essential cookies", "warrioriq_session keeps a signed-in account authenticated for up to 30 days. warrioriq_guest separates a temporary guest job from other browser sessions for up to 24 hours. warrioriq_oauth is a signed, HttpOnly cookie used for up to 10 minutes to protect an optional social sign-in attempt and is removed when the browser session ends. These cookies use Secure on HTTPS; the social-sign-in cookie uses SameSite=None in production so Apple can safely return a form POST, while the others use SameSite=Lax."),
            ("Preference storage", "WarriorIQ stores a first-party preference cookie recording Accept All, Reject Non-Essential or a custom selection. Signed-in choices are also linked to the account. Essential security and session storage remains separate and cannot be disabled while using authenticated or temporary private functions."),
            ("Analytics", "Google Analytics 4 is installed through a Google tag. The tag is not loaded at all until you choose Accept All or select analytics under Manage Preferences, and then only on WarriorIQ's public pages: the home page, pricing and the guides. It is never loaded on sign-in, account, upload, fight-selection, progress, report, replay, sharing or legal pages, so the addresses of those pages are never sent to Google. Reject Non-Essential keeps it off everywhere."),
            ("Advertising", "The same Google tag carries a Google Ads destination, which receives a page-view signal from those public pages once you have accepted analytics. Advertising storage, ad user data and ad personalisation are set to denied and are never granted by WarriorIQ, including under Accept All, so no advertising cookie is written and the signal is sent without personalisation. Fight footage and report contents are not sent to Google."),
            ("Managing storage", "You can remove cookies and local storage in browser settings. Removing an essential cookie can sign you out or make a temporary guest analysis inaccessible. The service must not use a cookie wall for functions that do not require optional tracking."),
        ],
    },
    "video-upload-policy": {
        "title": "Video Upload Policy",
        "description": "Rights, privacy and permission requirements for fight footage.",
        "intro": "Only upload footage that you are legally permitted to store and process with WarriorIQ.",
        "sections": [
            ("Your rights", "You must own the footage or have the necessary licence or permission to upload, privately store and analyse it. WarriorIQ receives only the limited permission needed to provide those requested functions; it does not automatically take ownership of the footage."),
            ("People shown", "You are responsible for an appropriate legal basis and any permissions required for identifiable athletes, referees, coaches, spectators or other people shown. If a minor appears, you must have appropriate parent or guardian permission and satisfy applicable safeguarding requirements."),
            ("Broadcast footage", "Do not upload an unauthorised UFC, GLORY, ONE, DAZN, television, streaming-platform or other copyrighted broadcast. Access to a stream or recording does not automatically grant a right to copy or process it."),
            ("Private processing", "Uploads, videos, reports and profiles are private by default. WarriorIQ does not create a public fight-video feed and does not provide automatic YouTube, broadcast or stream downloading."),
            ("Removal and complaints", "Rights holders can use the copyright-report form. The operator may restrict or remove reported content while investigating and may suspend repeat or serious violations under the Acceptable Use Policy."),
        ],
    },
    "sports-medical-disclaimer": {
        "title": "Sports and Medical Disclaimer",
        "description": "Training purpose and important health limits of WarriorIQ analysis.",
        "intro": "WarriorIQ is an AI-assisted combat-sports training and performance analytics platform.",
        "sections": [
            ("Training purpose", "Results are intended for training and educational review with a qualified coach. WarriorIQ is not an official fight judging authority and scores or statistics are estimates."),
            ("No medical advice", "WarriorIQ does not provide medical advice, diagnose concussion or injury, estimate brain damage or medical conditions, or determine whether an athlete is medically safe to continue fighting."),
            ("Seek qualified care", "Stop training and seek an appropriately qualified healthcare professional or emergency service for possible concussion, injury, severe symptoms or any health concern. Do not rely on a WarriorIQ report for a medical decision."),
            ("No diagnosis features", "Health-diagnosis features must not be added without separate medical, legal, privacy, safety and regulatory review."),
        ],
    },
    "acceptable-use": {
        "title": "Acceptable Use Policy",
        "description": "Safety, rights and fair-use rules for WarriorIQ uploads and accounts.",
        "intro": "Use WarriorIQ to improve legitimate combat-sports training—not to exploit, surveil or harm people.",
        "sections": [
            ("Upload responsibly", "Upload only footage you are authorised to use. Do not upload intimate, exploitative, abusive, stolen, secretly recorded or otherwise unlawful content. Footage involving a child must be managed by a parent or legal guardian where required, with the rights and documented lawful basis needed for recording, analysis and sharing."),
            ("Protect people", "Do not use output to stalk, identify, shame, discriminate against, threaten or make employment, insurance, credit, education, immigration, law-enforcement or other high-impact decisions about a person."),
            ("Protect the service", "Do not probe for vulnerabilities without written permission, evade quotas, access another account, automate abusive traffic, upload malware, interfere with analysis jobs or attempt to extract secrets or protected model assets."),
            ("Enforcement", "Suspected violations may be investigated and content or access may be restricted where proportionate. Illegal material and credible threats may be preserved or reported when the operator is legally required to do so."),
        ],
    },
    "refunds": {
        "title": "Refund and Cancellation Policy",
        "description": "WarriorIQ subscription renewal, cancellation, refund and withdrawal information.",
        "intro": "The checkout must show the exact recurring price, billing period, included usage and payment obligation before an order is placed.",
        "sections": [
            ("Cancellation", "A subscription can be cancelled through the configured billing portal or by contacting support. Cancellation stops the next renewal; access normally continues through the paid period unless the checkout terms say otherwise."),
            ("Refund requests", "Send the account email, charge date and reason to the configured support address. Requests are assessed under applicable consumer law and the terms displayed at purchase. This policy does not reduce mandatory refund, conformity, cooling-off or charge-dispute rights."),
            ("EU/EEA withdrawal", "Where a statutory withdrawal period applies, the checkout must explain it before purchase. Immediate digital-service performance and any request to begin during that period must use a separate, explicit acknowledgement; WarriorIQ must not assume a waiver merely because the user paid."),
            ("Service failures", "If a paid analysis fails before completion, its reserved analysis use is returned automatically. Billing refunds for prolonged outages, duplicate charges or materially unavailable service are handled separately from analysis credits."),
        ],
    },
    "eula": {
        "title": "End-User Licence Agreement",
        "description": "Licence terms for future installable WarriorIQ software.",
        "intro": "This EULA applies only when WarriorIQ is distributed as installable desktop or mobile software. The web service remains governed by the Terms of Service.",
        "sections": [
            ("Licence", "The operator grants the user a limited, revocable, non-exclusive, non-transferable licence to install and use one authorised copy for its intended fight-analysis purpose, subject to the selected plan and applicable store rules."),
            ("Restrictions", "Do not redistribute, rent, sublicense, defeat technical limits, remove notices, use the software maliciously or reverse engineer it except where applicable law expressly permits."),
            ("Updates and third-party components", "Security and compatibility updates may be required. Open-source and third-party components remain governed by their own licences and notices, which must ship with an installable release."),
            ("App stores and devices", "Apple, Google or another store may impose additional terms. The operator—not the store—is responsible for the app and support except where store terms state otherwise."),
            ("Termination", "The licence ends when these terms are materially breached or the user deletes the app, subject to mandatory rights. Terms that logically survive—such as ownership and lawful limitations—continue."),
        ],
    },
    "dmca": {
        "title": "Copyright and DMCA Policy",
        "description": "How to report copyright infringement and respond to a removal notice.",
        "intro": "WarriorIQ respects copyright. A valid notice should identify the work, the allegedly infringing material, contact details, good-faith and accuracy statements, and a physical or electronic signature.",
        "sections": [
            ("Send a notice", "Send a sufficiently detailed notice to the designated copyright contact in the Legal Notice. The operator may ask for missing information and may remove or restrict material when a valid notice is received."),
            ("Counter-notice", "A user who believes material was removed by mistake may submit identification of the removed material, a statement under penalty of perjury, consent to the appropriate court jurisdiction where legally required, contact details and a signature. The operator may restore material when the applicable process permits."),
            ("Repeat infringement", "Accounts of repeat infringers may be restricted or terminated in appropriate circumstances, while guarding against fraudulent notices and considering lawful exceptions."),
            ("Registration status", "Publishing a contact here does not register a U.S. DMCA designated agent. If the operator relies on the U.S. safe-harbour process, the real agent details must also be registered with the U.S. Copyright Office and kept current before public launch."),
        ],
    },
    "accessibility": {
        "title": "Accessibility Statement",
        "description": "WarriorIQ's accessibility target, current support and feedback route.",
        "intro": "WarriorIQ targets WCAG 2.2 Level AA and accessible e-commerce operation. Accessibility is an ongoing product requirement, not a one-time badge.",
        "sections": [
            ("What the interface supports", "Pages use semantic headings, labelled fields, keyboard-operable controls, visible focus, text alternatives, responsive layouts and status messages that do not rely only on colour. Video evidence should retain native playback controls."),
            ("Known limits", "Complex canvas-based fighter selection and skeleton overlays may be difficult with some assistive technology. The selection workflow needs continued testing with keyboard, zoom, screen readers, reduced motion and mobile devices before a public accessibility claim is final."),
            ("Feedback", "Report the page, device, browser, assistive technology and problem to the configured support email. WarriorIQ acknowledges accessibility requests and offers a reasonable alternative where it can."),
        ],
    },
    "ai-transparency": {
        "title": "AI Transparency Notice",
        "description": "How WarriorIQ uses automated analysis, its limits and automatic confidence controls.",
        "intro": "WarriorIQ uses computer vision and temporal models to track selected fighters and propose fight events. Customer reports are automatic, and outputs are labelled when they are preliminary or uncertain.",
        "sections": [
            ("What the AI does", "Local models estimate person boxes, pose, identity continuity, motion and action candidates. Ruleset logic filters legal techniques and an evidence layer decides what may be shown. Optional OpenAI identity recovery sends selected frames only after explicit upload-time opt-in and only when configured."),
            ("What the AI does not do", "It does not provide an official WAKO result, biometric identification of a real-world identity, medical advice, or a decision with legal or similarly significant effects. It can confuse fighters, limbs, techniques, timing, contact or ruleset outcomes."),
            ("User control", "Users choose Fighter A and Fighter B once, then the performance report runs automatically. No scorecard labelling or correction work is required. Users can replay supported evidence, and unsupported measurements remain hidden or clearly unavailable. Preliminary estimates never become verified facts merely because tracking coverage is high."),
            ("Quality and complaints", "Tracking coverage is a system observation metric, not a guarantee of correctness. Report material errors through support and include the analysis identifier rather than sending the original filename in public channels."),
        ],
    },
    "security": {
        "title": "Security Overview",
        "description": "WarriorIQ's current security controls and responsible disclosure route.",
        "intro": "Security details are stated narrowly so this page does not promise controls that are not implemented.",
        "sections": [
            ("Current application controls", "Passwords are salted and hashed; session tokens are stored as digests; account data and report access are owner-scoped; guest identifiers separate temporary sessions; uploads use generated storage names, extension allowlists, byte limits, file-signature checks and video decoding checks; sensitive responses disable caching; and security headers restrict framing, content types, browser permissions and content sources."),
            ("Abuse protection", "Requests are rate-limited per network address, with a readable page saying how long to wait, and repeated sign-in attempts from one address are refused for a while. The API schema and administration pages are not public."),
            ("Payments", "Card details are handled by configured Stripe-hosted checkout rather than stored by WarriorIQ. Stripe webhook signatures are verified and event identifiers are processed idempotently."),
            ("Responsible disclosure", "Send a concise vulnerability report to the configured support address, including reproduction steps and impact. Do not access other users' data, disrupt service or publish exploitable details before the operator has had a reasonable opportunity to investigate."),
        ],
    },
    "subprocessors": {
        "title": "Service Providers and Subprocessors",
        "description": "Which external providers may process WarriorIQ data and when.",
        # Built per request from this deployment's configuration by
        # subprocessor_sections(): the fixed list said "no WarriorIQ cloud host
        # is configured" on a site hosted on Render with analysis on Modal, and
        # carried "the operator must..." notes meant for whoever deployed it.
        "intro": "The providers below receive personal data from WarriorIQ, each for the purpose stated. The list follows this website's current configuration, so a provider appears here once it is switched on.",
        "sections": [],
    },
    "contact": {
        "title": "Contact and Complaints",
        "description": "How to reach WarriorIQ support, privacy, accessibility and copyright contacts.",
        # The second sentence used to assert that launch was blocked "until the
        # real operator and contact details are supplied". It was a fixed
        # string, so it said that whatever the configuration held - and it was
        # printed directly above a working mailto: address, telling the reader
        # no contact details existed while showing them one. Readiness is
        # already answered per request: the template renders a launch-aware
        # line below when launch_readiness() reports fields still missing, and
        # drops it once they are set. One place, and it tracks reality.
        "intro": "Each purpose below names the address to use.",
        "sections": [
            ("Product and account support", "Write to {support_email} for account access, billing, cancellation, accessibility help, analysis problems and general complaints. Include the analysis identifier when relevant, but do not send fight footage unless support specifically provides a secure channel."),
            ("Privacy rights", "Write to {privacy_email} for access, correction, deletion, restriction, objection, portability or consent-withdrawal requests. The operator may need proportionate information to verify the requester before disclosing personal data."),
            ("Copyright notices", "Write to {dmca_email} for infringement notices and counter-notices. The Copyright and DMCA Policy explains the information required and the limits of the published process."),
            ("Complaint handling", "WarriorIQ acknowledges complaints, investigates them fairly and explains the outcome where it lawfully can. If you are in the EU or the UK you can also complain to your data-protection authority about how your personal data is handled."),
        ],
    },
}


# What to say when an address is not configured.
#
# /contact used to read "Use the configured support email" in one section and
# "the configured privacy email" in the next, while the three real addresses
# appeared only in the operator block at the foot of the page. That is an
# unfilled template: the reader has to scroll past the whole document and then
# work out which of three addresses the section they were reading meant.
#
# The sections carry the address inline now. Where one is genuinely not set,
# the old phrasing is what remains - an empty mailto, or an invented address,
# would be worse than saying it is configured elsewhere. Nothing here makes up
# a contact detail that has not been supplied.
CONTACT_ADDRESS_FALLBACKS = {
    "support_email": "the configured support email",
    "privacy_email": "the configured privacy email",
    "dmca_email": "the configured copyright address",
}


# Words that name a protocol, a role or nothing at all rather than a company.
# QA, 2026-10-07: /subprocessors listed the email provider as "smtp".
PLACEHOLDER_NAMES = frozenset({
    "smtp", "smtps", "imap", "sendmail", "mail", "email", "e-mail", "mailer", "host", "hosting", "server",
    "localhost", "provider", "gpu", "worker", "cloud", "vps", "tbd", "todo", "tba", "changeme", "example",
    "n/a", "na", "none", "null", "unknown", "-", "?",
})
FREE_MAIL_DOMAINS = ("gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "yahoo.com", "icloud.com")


def is_placeholder_name(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return not text or text in PLACEHOLDER_NAMES or "example" in text or text.startswith(("your ", "the "))


def _named(value: str | None) -> bool:
    return not is_placeholder_name(value)


def public_config_problems(settings=None) -> list[str]:
    """What a public legal page would print that is a placeholder, not a fact.

    Run by tools/check_public_config.py (and tools/verify_project.py against a
    configured deployment), which fail on any of these.
    """
    cfg = settings or SETTINGS
    problems = []
    for label, value in (("WARRIORIQ_HOSTING_PROVIDER", cfg.hosting_provider),
                         ("WARRIORIQ_EMAIL_PROVIDER_NAME", cfg.email_provider_name),
                         ("WARRIORIQ_ANALYSIS_PROVIDER_NAME", cfg.analysis_provider_name)):
        if str(value or "").strip() and is_placeholder_name(value):
            problems.append(f"{label} is {value!r}, which is not a company name")
    if cfg.email_provider and not str(cfg.email_provider_name or "").strip():
        problems.append("email is switched on (WARRIORIQ_EMAIL_PROVIDER) but WARRIORIQ_EMAIL_PROVIDER_NAME, "
                        "the company that delivers it, is empty")
    if not str(cfg.hosting_provider or "").strip():
        problems.append("WARRIORIQ_HOSTING_PROVIDER is empty, so /subprocessors cannot name the host")
    for key in ("support_email", "privacy_email"):
        address = str(getattr(cfg, key, "") or "").strip()
        if not address:
            problems.append(f"WARRIORIQ_{key.upper()} is empty, so legal pages print a placeholder for it")
        elif address.lower().rsplit("@", 1)[-1] in FREE_MAIL_DOMAINS:
            problems.append(f"WARRIORIQ_{key.upper()} is a {address.rsplit('@', 1)[-1]} address, not one on "
                            "WarriorIQ's own domain")
    return problems


def subprocessor_sections() -> list[tuple[str, str]]:
    """Who receives personal data from this deployment, and for what.

    Read from configuration rather than written out, so the page names the
    host and analysis provider actually in use and nothing that is switched
    off. Only providers that receive personal data are listed.
    """
    import os

    sections: list[tuple[str, str]] = []
    if SETTINGS.hosting_provider:
        sections.append((f"{SETTINGS.hosting_provider} (website hosting and storage)", (
            f"{SETTINGS.hosting_provider} hosts this website. Accounts, the database, uploaded fight videos and "
            "finished reports are stored on its servers. Saved fight videos are deleted after "
            f"{SETTINGS.saved_video_retention_days} days.")))
    else:
        sections.append(("Website hosting and storage", (
            "Accounts, the database, uploaded fight videos and finished reports are stored on the server that "
            f"runs this website. Saved fight videos are deleted after {SETTINGS.saved_video_retention_days} days.")))
    if "modal.run" in SETTINGS.worker_wake_url.lower():
        sections.append(("Modal (fight analysis)", (
            "Modal provides the cloud graphics card that analyses each fight. For one analysis it receives the "
            "fight video and the two fighter boxes you drew, returns the measurements to this website, and the "
            "video and working files are deleted from it when the run ends.")))
    elif SETTINGS.analysis_worker_mode == "remote" and _named(SETTINGS.analysis_provider_name):
        name = SETTINGS.analysis_provider_name
        sections.append((f"{name} (fight analysis)", (
            f"{name} provides the machine that analyses each fight. For one analysis it downloads the fight "
            "video and the two fighter boxes from this website and deletes them when the run ends.")))
    elif SETTINGS.analysis_worker_mode == "remote":
        sections.append(("Analysis computer", (
            "Fights are analysed on a separate analysis computer run for WarriorIQ. For one analysis it downloads "
            "the fight video and the two fighter boxes from this website and deletes them when the run ends.")))
    if SETTINGS.analytics_measurement_id or SETTINGS.gtm_container_id:
        sections.append(("Google (analytics and advertising measurement)", (
            "Google receives website usage data through a Google tag, which loads only after you accept analytics "
            "and only on WarriorIQ's public pages (home, pricing and guides): analytics events, and a page-view "
            "signal to Google Ads sent without personalisation. It never loads on sign-in, account, upload, report, "
            "replay or sharing pages, so their addresses are never sent. Advertising storage and ad "
            "personalisation are refused at all times. Fight footage and report contents are never sent.")))
    if SETTINGS.email_provider and _named(SETTINGS.email_provider_name):
        name = SETTINGS.email_provider_name
        sections.append((f"{name} (email)", (
            f"{name} delivers WarriorIQ's account emails, such as sign-in and verification "
            "messages, and so receives your email address and the message.")))
    elif SETTINGS.email_provider:
        # Email is switched on but its company is not named. Never the
        # transport ("smtp"); tools/check_public_config.py fails on this.
        sections.append(("Email delivery", (
            "An email delivery service sends WarriorIQ's account emails, such as sign-in and verification "
            "messages, and so receives your email address and the message.")))
    sign_in = [name for name, configured in (
        ("Google", SETTINGS.google_client_id and SETTINGS.google_client_secret),
        ("Meta/Facebook", SETTINGS.facebook_client_id and SETTINGS.facebook_client_secret),
        ("Microsoft", SETTINGS.microsoft_client_id and SETTINGS.microsoft_client_secret),
        ("GitHub", SETTINGS.github_client_id and SETTINGS.github_client_secret),
    ) if configured and SETTINGS.oauth_state_secret]
    if sign_in:
        names = sign_in[0] if len(sign_in) == 1 else ", ".join(sign_in[:-1]) + " and " + sign_in[-1]
        sections.append(("Sign-in providers", (
            f"{names} {'is' if len(sign_in) == 1 else 'are'} offered for signing in. A provider receives data only "
            "when you choose its button; WarriorIQ receives a provider identifier and your email or display name, "
            "and does not keep the provider's access tokens.")))
    if SETTINGS.payments_enabled and os.getenv("STRIPE_SECRET_KEY", "").strip():
        sections.append(("Stripe (payments)", (
            "Stripe runs checkout for paid plans and receives your email and billing details. WarriorIQ never "
            "sees or stores full card details.")))
    if os.getenv("OPENAI_API_KEY", "").strip():
        sections.append(("OpenAI (optional identity recovery)", (
            "Only if you switch on identity recovery for an upload, selected frames of that fight are sent to "
            "OpenAI to help tell Fighter A from Fighter B when tracking is unsure.")))
    sections.append(("Changes to this list", (
        "A new provider is added to this page before it receives personal data, with an updated policy "
        "version.")))
    return sections


def resolve_document(slug: str) -> dict | None:
    """A published legal document with its contact placeholders filled in.

    Filled per call rather than baked in at import, so that changing an address
    needs only the restart every other setting needs, and so a test that
    patches SETTINGS sees what it patched.
    """
    document = LEGAL_DOCUMENTS.get(slug)
    if document is None:
        return None
    addresses = {
        key: (getattr(SETTINGS, key, "") or fallback)
        for key, fallback in CONTACT_ADDRESS_FALLBACKS.items()
    }
    # Only text that actually carries a placeholder is formatted, so a stray
    # brace in ordinary prose can never raise on a legal page.
    fill = lambda text: text.format(**addresses) if "{" in text else text
    sections = subprocessor_sections() if slug == "subprocessors" else document["sections"]
    return {
        **document,
        "intro": fill(document.get("intro", "")),
        "sections": [(heading, fill(body)) for heading, body in sections],
    }

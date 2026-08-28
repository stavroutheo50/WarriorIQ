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
            "name": SETTINGS.operator_name or "Not configured",
            "address": SETTINGS.operator_address or "Not configured",
            "registration": SETTINGS.operator_registration or "Not configured",
            "vat": SETTINGS.operator_vat or "Not configured",
            "country": SETTINGS.governing_country or "Not configured",
            "support_email": SETTINGS.support_email or "Not configured",
            "privacy_email": SETTINGS.privacy_email or "Not configured",
            "dmca_email": SETTINGS.dmca_email or "Not configured",
            "dmca_agent_name": SETTINGS.dmca_agent_name or "Not configured",
        },
    }


LEGAL_DOCUMENTS = {
    "terms": {
        "title": "Terms of Service",
        "description": "The rules for using WarriorIQ's fight-analysis website and athlete workspace.",
        "intro": "These Terms govern the WarriorIQ website and analysis service and take effect when a user accepts the policy version shown below.",
        "sections": [
            ("Who may use WarriorIQ", "The initial public launch permits standard account registration only for people aged 18 or older. Fight footage may include junior athletes only when the uploader has every required right and appropriate parent or guardian permission. Junior accounts are not enabled yet; they require a separately reviewed age-assurance, verifiable guardian-approval, privacy and safeguarding programme."),
            ("Your footage and permissions", "You keep ownership of your footage. You give the operator a limited licence to store, decode, process, display and delete it only as needed to provide the service. Before uploading, you must have the rights to the recording and an appropriate legal basis or permission for every identifiable person shown. A checkbox does not replace consent or another legal basis required by local law."),
            ("Analysis limitations", "WarriorIQ produces training and coaching support, not an official judging decision, medical assessment, safety guarantee or substitute for a qualified coach. Automated labels, identities and estimated scorecards can be wrong. Confidence labels and evidence replay must be considered before acting on a result."),
            ("Accounts and security", "Provide accurate account information, protect your password, and tell support about suspected unauthorised access. A parent or legal guardian managing a child's workspace is responsible for its settings, uploads and sharing choices unless the law says otherwise."),
            ("Plans, billing and cancellation", "Paid plans renew for the period shown at checkout until cancelled. The price, taxes, analysis allowance, report depth and renewal terms must be presented before payment. Cancellation stops future renewal and does not erase legal refund or withdrawal rights. The Refund and Cancellation Policy forms part of these Terms."),
            ("Acceptable use", "Do not upload unlawful, stolen, abusive or secretly recorded content; identify or harass people; bypass access controls or allowances; scrape the service; reverse engineer protected components; distribute malware; or use the service to make high-impact decisions about another person. The Acceptable Use Policy gives more detail."),
            ("Suspension and termination", "Access may be restricted when reasonably necessary to protect users, the service, legal rights or security. Where practical and lawful, the operator should explain the reason and give a route to appeal. Users may delete their account from the Athlete Profile."),
            ("Intellectual property", "WarriorIQ branding, interface and original software are protected by applicable intellectual-property law. These Terms do not transfer them. User footage and lawful feedback remain subject to the rights described in these Terms and the Privacy Policy."),
            ("Disclaimers and liability", "The service is provided with the care required by applicable law, but uninterrupted operation and perfect analysis are not promised. Nothing in these Terms excludes rights or liability that cannot legally be excluded, including mandatory consumer protections."),
            ("Changes and contact", "Material changes require a new policy version and renewed acceptance when appropriate. The configured operator and governing jurisdiction appear in the Legal Notice. Questions can be sent to the configured support address."),
        ],
    },
    "cookies": {
        "title": "Cookie Policy",
        "description": "What WarriorIQ stores in the browser and how cookie preferences work.",
        "intro": "WarriorIQ uses essential browser storage for security and private job access. Optional advertising or cross-site tracking is not installed in this build.",
        "sections": [
            ("Essential cookies", "warrioriq_session keeps a signed-in account authenticated for up to 30 days. warrioriq_guest separates a temporary guest job from other browser sessions for up to 24 hours. Both are HttpOnly, SameSite=Lax, and Secure when the site uses HTTPS."),
            ("Preference storage", "WarriorIQ stores a first-party preference cookie recording Accept All, Reject Non-Essential or a custom selection. Signed-in choices are also linked to the account. Essential security and session storage remains separate and cannot be disabled while using authenticated or temporary private functions."),
            ("Analytics", "No third-party analytics tracker is installed in this build. If product analytics is added later, non-essential storage must remain off until the user makes a clear choice, and this policy and the preference controls must be updated first."),
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
            ("Feedback", "Report the page, device, browser, assistive technology and problem to the configured support email. The operator should acknowledge accessibility requests and offer a reasonable alternative when possible."),
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
            ("Current application controls", "Passwords are salted and hashed; session tokens are stored as digests; account data and report access are owner-scoped; guest identifiers separate temporary sessions; uploads use generated storage names, extension allowlists, byte limits and video decoding checks; sensitive responses disable caching; and security headers restrict framing, content types, browser permissions and content sources."),
            ("Payments", "Card details are handled by configured Stripe-hosted checkout rather than stored by WarriorIQ. Stripe webhook signatures are verified and event identifiers are processed idempotently."),
            ("Responsible disclosure", "Send a concise vulnerability report to the configured support address, including reproduction steps and impact. Do not access other users' data, disrupt service or publish exploitable details before the operator has had a reasonable opportunity to investigate."),
            ("Production obligations", "A public operator still needs managed secrets, encrypted backups, access logging, patching, rate limits, malware scanning, incident response, vendor review, tested restoration and jurisdiction-appropriate breach procedures. Launch readiness does not claim those operational tasks are complete."),
        ],
    },
    "subprocessors": {
        "title": "Service Providers and Subprocessors",
        "description": "Which external providers may process WarriorIQ data and when.",
        "intro": "This list describes integrations present in the current WarriorIQ code. The final operator must update it to match the actual hosting region, vendor contracts and production configuration.",
        "sections": [
            ("OpenAI", "Optional identity-recovery provider. Selected fight frames are sent only when the user enables the feature and an API key is configured. Purpose: resolving Fighter A/B continuity when local tracking is uncertain. The operator must document the contracted entity, processing location, retention controls and transfer mechanism before production use."),
            ("Stripe", "Optional payment provider. When paid plans are enabled, Stripe receives checkout, account email and billing information and returns signed payment events. WarriorIQ does not store full card details. The contracted Stripe entity and region depend on the operator's account."),
            ("Hosting and storage", "This build processes and stores files on the computer running it. No WarriorIQ cloud host is configured in the code. A public host, CDN, database, email or monitoring vendor must be added to this notice before it receives personal data."),
            ("Change notice", "Material provider changes should be published before they take effect where required, with an updated policy date and a way for affected customers to raise data-protection concerns."),
        ],
    },
    "contact": {
        "title": "Contact and Complaints",
        "description": "How to reach WarriorIQ support, privacy, accessibility and copyright contacts.",
        "intro": "Use the configured address for the purpose below. Public launch remains blocked until the real operator and contact details are supplied in the launch configuration.",
        "sections": [
            ("Product and account support", "Use the configured support email for account access, billing, cancellation, accessibility help, analysis problems and general complaints. Include the analysis identifier when relevant, but do not send fight footage unless support specifically provides a secure channel."),
            ("Privacy rights", "Use the configured privacy email for access, correction, deletion, restriction, objection, portability or consent-withdrawal requests. The operator may need proportionate information to verify the requester before disclosing personal data."),
            ("Copyright notices", "Use the configured copyright address for infringement notices and counter-notices. The Copyright and DMCA Policy explains the information required and the limits of the published process."),
            ("Complaint handling", "The operator should acknowledge complaints, investigate them fairly, explain the outcome where lawful, and identify any independent regulator, consumer-dispute route or appeal mechanism required in the user's jurisdiction."),
        ],
    },
}

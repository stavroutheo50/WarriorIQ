from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from authlib.integrations.starlette_client import OAuth

from core.config import SETTINGS


@dataclass(frozen=True)
class SocialIdentity:
    provider: str
    subject: str
    email: str | None
    display_name: str | None
    email_verified: bool = False


# Apple is deliberately absent. Sign in with Apple needs the Apple Developer
# Program ($99/year, and enrolment requires the legal age of majority), and its
# "client secret" is a JWT that expires within six months, so it would fail
# silently twice a year unless minted on every request. Re-adding it means
# restoring this label, the registration block, and the icon in auth.html.
PROVIDER_LABELS = {
    "google": "Google",
    "facebook": "Facebook",
    "microsoft": "Microsoft",
    "github": "GitHub",
}

# Where each provider's consent page lives.
#
# Pressing a social button submits a form, and the reply is a redirect to the
# provider. Content-Security-Policy's form-action is enforced across that
# redirect, so a policy of 'self' blocks the hop to the provider and Chrome
# aborts the submission - the button had already been switched to "Working..."
# by then, so the page sat there apparently loading for ever. Only the origins
# of providers that are actually configured are ever added to the policy.
AUTHORIZE_ORIGINS = {
    "google": "https://accounts.google.com",
    "facebook": "https://www.facebook.com",
    "microsoft": "https://login.microsoftonline.com",
    "github": "https://github.com",
}

# Every outbound call to a provider is made while somebody waits on a button.
# Without a ceiling, a host that cannot reach the provider turns that button
# into an infinite spinner instead of an error.
OUTBOUND_TIMEOUT_SECONDS = 10.0


class SocialAuthRegistry:
    """Configured OAuth/OIDC providers without retaining provider tokens."""

    def __init__(self) -> None:
        self.oauth = OAuth()
        self._enabled: set[str] = set()
        if not SETTINGS.oauth_state_secret:
            return
        self._register_oidc(
            "google", SETTINGS.google_client_id, SETTINGS.google_client_secret,
            "https://accounts.google.com/.well-known/openid-configuration",
        )
        self._register_oidc(
            "microsoft", SETTINGS.microsoft_client_id, SETTINGS.microsoft_client_secret,
            "https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration",
        )
        if SETTINGS.facebook_client_id and SETTINGS.facebook_client_secret:
            self.oauth.register(
                name="facebook",
                client_id=SETTINGS.facebook_client_id,
                client_secret=SETTINGS.facebook_client_secret,
                access_token_url="https://graph.facebook.com/oauth/access_token",
                authorize_url="https://www.facebook.com/dialog/oauth",
                api_base_url="https://graph.facebook.com/",
                client_kwargs={"scope": "email", "timeout": OUTBOUND_TIMEOUT_SECONDS},
            )
            self._enabled.add("facebook")
        # GitHub is not an OpenID Connect provider: there is no id_token and no
        # discovery document, so the identity is read from its own user API.
        # Its endpoints are fixed, which means pressing this button needs no
        # network call before the redirect at all.
        if SETTINGS.github_client_id and SETTINGS.github_client_secret:
            self.oauth.register(
                name="github",
                client_id=SETTINGS.github_client_id,
                client_secret=SETTINGS.github_client_secret,
                access_token_url="https://github.com/login/oauth/access_token",
                authorize_url="https://github.com/login/oauth/authorize",
                api_base_url="https://api.github.com/",
                client_kwargs={"scope": "read:user user:email", "timeout": OUTBOUND_TIMEOUT_SECONDS},
            )
            self._enabled.add("github")

    def _register_oidc(self, name: str, client_id: str, client_secret: str, metadata_url: str) -> None:
        if not client_id or not client_secret:
            return
        self.oauth.register(
            name=name,
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url=metadata_url,
            # The scope and PKCE method are the provider's contract. The
            # timeout is ours, and it is the important one: pressing this
            # button makes the web server fetch the provider's discovery
            # document, and the web server is a shared host whose outbound
            # HTTPS can be blocked or slow. Without a timeout that fetch never
            # returns, the POST never answers, and the button spins for ever
            # with nothing shown to the person pressing it - which is exactly
            # how this failed in production. Ten seconds is far longer than a
            # reachable provider needs and short enough to report as an error.
            client_kwargs={
                "scope": "openid email profile",
                "code_challenge_method": "S256",
                "timeout": OUTBOUND_TIMEOUT_SECONDS,
            },
        )
        self._enabled.add(name)

    @property
    def provider_buttons(self) -> list[dict[str, str]]:
        return [
            {"key": key, "label": PROVIDER_LABELS[key]}
            for key in PROVIDER_LABELS
            if key in self._enabled
        ]

    @property
    def form_action_origins(self) -> list[str]:
        """Origins a sign-in form is allowed to be redirected to."""
        return [AUTHORIZE_ORIGINS[key] for key in sorted(self._enabled) if key in AUTHORIZE_ORIGINS]

    def is_enabled(self, provider: str) -> bool:
        return provider in self._enabled

    def client(self, provider: str):
        return self.oauth.create_client(provider) if self.is_enabled(provider) else None

    async def identity_from_token(self, provider: str, client, token: dict[str, Any]) -> SocialIdentity:
        if provider == "facebook":
            response = await client.get("me", token=token, params={"fields": "id,name,email"})
            response.raise_for_status()
            payload = response.json()
            subject = str(payload.get("id") or "")
            email = payload.get("email")
            name = payload.get("name")
            email_verified = False
        elif provider == "github":
            response = await client.get("user", token=token)
            response.raise_for_status()
            payload = response.json()
            subject = str(payload.get("id") or "")
            name = payload.get("name") or payload.get("login")
            email = payload.get("email")
            email_verified = False
            if not email:
                # A GitHub profile email is public only if the account chose to
                # publish it, so the primary address usually has to be asked
                # for separately. Failing that, the account is still usable -
                # it just has no address attached.
                try:
                    listing = await client.get("user/emails", token=token)
                    listing.raise_for_status()
                    for entry in listing.json():
                        if entry.get("primary"):
                            email = entry.get("email")
                            email_verified = bool(entry.get("verified"))
                            break
                except Exception:                       # noqa: BLE001
                    email = None
        else:
            payload = dict(token.get("userinfo") or {})
            if not payload:
                payload = dict(await client.userinfo(token=token))
            subject = str(payload.get("sub") or "")
            email = payload.get("email") or (
                payload.get("preferred_username") if provider == "microsoft" else None
            )
            name = payload.get("name")
            raw_verified = payload.get("email_verified", False)
            email_verified = raw_verified is True or str(raw_verified).lower() == "true"
        if not subject or len(subject) > 512:
            raise ValueError("The identity provider did not return a valid account identifier.")
        normalized_email = str(email).strip().lower() if email else None
        return SocialIdentity(
            provider=provider,
            subject=subject,
            email=normalized_email,
            display_name=str(name).strip()[:120] if name else None,
            email_verified=email_verified,
        )


SOCIAL_AUTH = SocialAuthRegistry()

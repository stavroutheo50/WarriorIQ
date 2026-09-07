"""A TestClient that carries the CSRF token, the way a real browser does.

Adding CSRF protection turned fifty existing tests red at once, and every one
of those failures was correct: they posted without a token, which is exactly
what the new gate refuses. The wrong fix would have been to exempt the routes
they touch. The right one is to make the test client behave like the browser
it stands in for - a browser gets the token from the page it is already on, so
this fetches one when it does not have it and returns it on every unsafe
request.

`CsrfTests` deliberately does not use this. It imports the plain client inside
its own setUp so that it still sees a request with no token, which is the thing
it is there to test.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

CSRF_COOKIE = "warrioriq_csrf"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# Any page issues a token; this one is public, cheap, and needs no account.
TOKEN_SOURCE = "/login"


class BrowserClient(TestClient):
    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        if str(method).upper() in UNSAFE_METHODS:
            token = self.cookies.get(CSRF_COOKIE)
            if not token:
                super().request("GET", TOKEN_SOURCE)
                token = self.cookies.get(CSRF_COOKIE)
            if token:
                headers = dict(kwargs.get("headers") or {})
                # setdefault, so a test that sets its own token to prove a
                # forgery is refused still gets the value it chose.
                headers.setdefault("X-CSRF-Token", token)
                kwargs["headers"] = headers
        return super().request(method, url, *args, **kwargs)

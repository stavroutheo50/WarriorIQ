"""Fixes from the 2026-11-09 site walkthrough."""

from __future__ import annotations

import unittest
import uuid

from browser_client import BrowserClient

import app.main as webapp
from core.auth import register


class AccountLoginLimitTests(unittest.TestCase):
    def setUp(self):
        webapp._rate_windows.clear()
        self.client = BrowserClient(webapp.app)
        self.email = f"limit-{uuid.uuid4().hex[:8]}@example.com"
        register(self.email, "Strong-Local-Password")

    def tearDown(self):
        webapp._rate_windows.clear()

    def attempt(self, email, password="wrong-password"):
        return self.client.post("/login", data={"email": email, "password": password, "next_path": "/camp"},
                                follow_redirects=False)

    def test_one_account_meets_a_limit_whatever_address_the_guesses_come_from(self):
        for _ in range(webapp.LOGIN_ATTEMPTS_PER_ACCOUNT):
            self.assertEqual(self.attempt(self.email).status_code, 400)
            webapp._rate_windows.pop("login:testclient", None)      # as if each came from a new address
        refused = self.attempt(self.email, "Strong-Local-Password")
        self.assertEqual(refused.status_code, 429)
        self.assertIn("Retry-After", refused.headers)

    def test_an_unregistered_email_is_limited_the_same_way(self):
        nobody = f"nobody-{uuid.uuid4().hex[:8]}@example.com"
        for _ in range(webapp.LOGIN_ATTEMPTS_PER_ACCOUNT):
            self.assertEqual(self.attempt(nobody).status_code, 400)
        self.assertEqual(self.attempt(nobody).status_code, 429)

    def test_other_accounts_are_not_affected(self):
        for _ in range(webapp.LOGIN_ATTEMPTS_PER_ACCOUNT):
            self.attempt(self.email)
        webapp._rate_windows.pop("login:testclient", None)
        other = f"other-{uuid.uuid4().hex[:8]}@example.com"
        register(other, "Strong-Local-Password")
        self.assertEqual(self.attempt(other, "Strong-Local-Password").status_code, 303)


class InvalidRequestTests(unittest.TestCase):
    def test_an_invalid_api_request_explains_itself_in_a_sentence(self):
        client = BrowserClient(webapp.app)
        email = f"invalid-{uuid.uuid4().hex[:8]}@example.com"
        register(email, "Strong-Local-Password")
        client.post("/login", data={"email": email, "password": "Strong-Local-Password", "next_path": "/camp"})
        response = client.post("/api/start/does-not-exist", json={"fighter_a_box": "not a box"})
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["detail"], webapp.INVALID_REQUEST_DETAIL)
        self.assertIsInstance(body["errors"], list)

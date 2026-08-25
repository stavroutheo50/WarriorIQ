from __future__ import annotations

import os

from core.config import SETTINGS


PLANS = {
    "free": {
        "label": "Starter",
        "price": "€0",
        "period": "forever",
        "description": "A simple daily check-in for athletes getting started.",
        "limit_label": "1 analysis every day",
        "report_label": "Compact report",
        "features": ["1 analysis per day", "Compact performance summary", "3 verified key moments", "1 coaching priority"],
        "daily_limit": 1,
        "monthly_limit": None,
        "unlimited": False,
        "report_tier": "compact",
        "evidence_limit": 3,
        "coaching_items": 1,
        "training_items": 0,
        "can_share": False,
        "can_correct": False,
        "credits": 0,
        "stripe_price_id": "",
        "mode": "none",
    },
    "athlete": {
        "label": "Athlete",
        "price": "€9.99",
        "period": "per month",
        "description": "For athletes reviewing training and competition every day.",
        "limit_label": "3 analyses every day",
        "report_label": "Expanded report",
        "features": ["3 analyses per day", "Expanded performance report", "Scorecard and core coaching", "Up to 8 verified key moments", "Private report sharing"],
        "daily_limit": 3,
        "monthly_limit": None,
        "unlimited": False,
        "report_tier": "expanded",
        "evidence_limit": 8,
        "coaching_items": 2,
        "training_items": 2,
        "can_share": True,
        "can_correct": False,
        "credits": 0,
        "stripe_price_id": os.getenv("STRIPE_PRICE_ATHLETE", ""),
        "mode": "subscription",
    },
    "athlete_pro": {
        "label": "Athlete Pro",
        "price": "€24.99",
        "period": "per month",
        "description": "The complete WarriorIQ experience for active competitors.",
        "limit_label": "10 analyses every day",
        "report_label": "Complete report",
        "features": ["10 analyses per day", "Complete performance analysis", "Full coach report and training plan", "Full evidence replay and legality review", "Fight comparisons and corrections"],
        "daily_limit": 10,
        "monthly_limit": None,
        "unlimited": False,
        "report_tier": "full",
        "evidence_limit": None,
        "coaching_items": None,
        "training_items": None,
        "can_share": True,
        "can_correct": True,
        "highlight": "Most flexible",
        "credits": 0,
        "stripe_price_id": os.getenv("STRIPE_PRICE_ATHLETE_PRO", os.getenv("STRIPE_PRICE_PRO", "")),
        "mode": "subscription",
    },
    "coach": {
        "label": "Coach",
        "price": "€59.99",
        "period": "per month",
        "description": "For coaches turning complete fight evidence into athlete work.",
        "limit_label": "30 analyses every month",
        "report_label": "Complete report",
        "features": ["30 analyses per month", "Complete performance analysis", "Coach assignments", "Athlete-ready private sharing", "Fight library and comparisons"],
        "daily_limit": None,
        "monthly_limit": 30,
        "unlimited": False,
        "report_tier": "full",
        "evidence_limit": None,
        "coaching_items": None,
        "training_items": None,
        "can_share": True,
        "can_correct": True,
        "credits": 30,
        "stripe_price_id": os.getenv("STRIPE_PRICE_COACH", ""),
        "mode": "subscription",
    },
    "gym": {
        "label": "Gym",
        "price": "€89.99",
        "period": "per month",
        "description": "The unlimited WarriorIQ suite for busy gyms and fight teams.",
        "limit_label": "Unlimited analyses",
        "report_label": "Complete report",
        "features": ["Unlimited fight analyses", "Everything in the Coach plan", "Complete reports for every fight", "Coach assignments and private sharing", "Saved fight library and comparisons", "Priority gym onboarding"],
        "daily_limit": None,
        "monthly_limit": None,
        "unlimited": True,
        "report_tier": "full",
        "evidence_limit": None,
        "coaching_items": None,
        "training_items": None,
        "can_share": True,
        "can_correct": True,
        "highlight": "Best for teams",
        "credits": 0,
        "stripe_price_id": os.getenv("STRIPE_PRICE_GYM", ""),
        "mode": "subscription",
    },
}


def plan_for_key(plan_key: str | None) -> dict:
    """Return a complete, safe entitlement contract for an account plan."""
    return PLANS.get(str(plan_key or "free").lower(), PLANS["free"])


def create_checkout(plan_key: str, success_url: str, cancel_url: str, account_id: int, email: str) -> str:
    if not SETTINGS.payments_enabled:
        raise RuntimeError("Payments are disabled. Set WARRIORIQ_PAYMENTS=1 after configuring a lawful Stripe account.")
    plan = PLANS.get(plan_key)
    if not plan or plan.get("mode") not in {"payment", "subscription"} or not plan["stripe_price_id"]:
        raise RuntimeError("This plan does not have a Stripe price ID configured.")
    secret = os.getenv("STRIPE_SECRET_KEY", "")
    if not secret:
        raise RuntimeError("STRIPE_SECRET_KEY is not configured.")
    import stripe

    stripe.api_key = secret
    session = stripe.checkout.Session.create(
        mode=plan["mode"],
        line_items=[{"price": plan["stripe_price_id"], "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        customer_email=email,
        client_reference_id=str(account_id),
        metadata={"warrioriq_account_id": str(account_id), "warrioriq_plan": plan_key},
    )
    return str(session.url)


def verify_webhook(payload: bytes, signature: str):
    secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    if not SETTINGS.payments_enabled or not secret:
        raise RuntimeError("Stripe webhooks are not configured.")
    import stripe

    return stripe.Webhook.construct_event(payload, signature, secret)

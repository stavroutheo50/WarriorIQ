from __future__ import annotations

import os

from core.config import SETTINGS


# How many fighters a workspace may hold, and who each plan is for.
#
# This is the line between the two products. An athlete analyses one person -
# themselves - so their plans hold one fighter and are priced for one person. A
# coach analyses a squad, and what they are buying is room for more names on
# the roster, because that is what decides how much work the account does.
# roster_limit of None means no ceiling.
#
# audience decides which plans a workspace is shown: a coach is not offered a
# one-fighter plan, and an athlete is not sold seats they will never use.
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
        "roster_limit": 1,
        "audience": "athlete",
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
        "roster_limit": 1,
        "audience": "athlete",
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
        "roster_limit": 1,
        "audience": "athlete",
        "stripe_price_id": os.getenv("STRIPE_PRICE_ATHLETE_PRO", os.getenv("STRIPE_PRICE_PRO", "")),
        "mode": "subscription",
    },
    # The coach ladder. Priced by how many fighters the roster holds, because
    # that is the thing a coach is actually buying and the thing that decides
    # how much analysis the account does. Everything above the seat count is
    # identical - a coach with five fighters needs the same reports as one with
    # thirty, just fewer of them.
    "coach_5": {
        "label": "Coach 5",
        "price": "€29.99",
        "period": "per month",
        "description": "For a corner working with a handful of fighters.",
        "limit_label": "Up to 5 fighters",
        "report_label": "Complete report",
        "features": ["Up to 5 fighters on the roster", "10 analyses per day",
                     "Complete report for every fighter", "Squad view and per-fighter trends",
                     "Full evidence replay and legality review"],
        "daily_limit": 10,
        "monthly_limit": None,
        "unlimited": False,
        "report_tier": "full",
        "evidence_limit": None,
        "coaching_items": None,
        "training_items": None,
        "can_share": True,
        "can_correct": True,
        "credits": 0,
        "roster_limit": 5,
        "audience": "coach",
        "stripe_price_id": os.getenv("STRIPE_PRICE_COACH_5", ""),
        "mode": "subscription",
    },
    "coach_15": {
        "label": "Coach 15",
        "price": "€54.99",
        "period": "per month",
        "description": "For a club squad training through a season.",
        "limit_label": "Up to 15 fighters",
        "report_label": "Complete report",
        "features": ["Up to 15 fighters on the roster", "Everything in Coach 5",
                     "Squad view and per-fighter trends", "Fight comparisons across the roster"],
        "daily_limit": 20,
        "monthly_limit": None,
        "unlimited": False,
        "report_tier": "full",
        "evidence_limit": None,
        "coaching_items": None,
        "training_items": None,
        "can_share": True,
        "can_correct": True,
        "highlight": "Most clubs",
        "credits": 0,
        "roster_limit": 15,
        "audience": "coach",
        "stripe_price_id": os.getenv("STRIPE_PRICE_COACH_15", ""),
        "mode": "subscription",
    },
    "coach_30": {
        "label": "Coach 30",
        "price": "€74.99",
        "period": "per month",
        "description": "For a full team with several coaches working from it.",
        "limit_label": "Up to 30 fighters",
        "report_label": "Complete report",
        "features": ["Up to 30 fighters on the roster", "Everything in Coach 15",
                     "40 analyses per day"],
        "daily_limit": 40,
        "monthly_limit": None,
        "unlimited": False,
        "report_tier": "full",
        "evidence_limit": None,
        "coaching_items": None,
        "training_items": None,
        "can_share": True,
        "can_correct": True,
        "credits": 0,
        "roster_limit": 30,
        "audience": "coach",
        "stripe_price_id": os.getenv("STRIPE_PRICE_COACH_30", ""),
        "mode": "subscription",
    },
    "gym": {
        "label": "Gym",
        "price": "€89.99",
        "period": "per month",
        "description": "The unlimited WarriorIQ suite for busy gyms and fight teams.",
        "limit_label": "Unlimited fighters",
        "report_label": "Complete report",
        "features": ["Unlimited fight analyses", "Everything in Athlete Pro", "Complete reports for every fight", "Coach assignments and private sharing", "Saved fight library and comparisons", "Priority gym onboarding"],
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
        "roster_limit": None,
        "audience": "coach",
        "stripe_price_id": os.getenv("STRIPE_PRICE_GYM", ""),
        "mode": "subscription",
    },
}


# The Coach plan was withdrawn. Anyone already carrying that key keeps
# equivalent access rather than silently dropping to the free tier, which is
# what an unknown key would otherwise do.
LEGACY_PLAN_ALIASES = {"coach": "athlete_pro"}


def plan_for_key(plan_key: str | None) -> dict:
    """Return a complete, safe entitlement contract for an account plan."""
    key = str(plan_key or "free").lower()
    return PLANS.get(LEGACY_PLAN_ALIASES.get(key, key), PLANS["free"])


def effective_plan_key(plan: str | None, plan_override: str | None, email: str | None = None) -> str:
    """Resolve the plan an account actually holds.

    A complimentary grant is an operator decision recorded in configuration
    rather than in the accounts table, so it survives a database restore and
    needs no manual row edit on the live host. It outranks the stored plan
    because it is issued deliberately.
    """
    granted = SETTINGS.complimentary_plans.get(str(email or "").strip().lower())
    if granted and granted in PLANS:
        return granted
    return str(plan_override or plan or "free")


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


def cancel_subscription_at_period_end(subscription_id: str) -> dict:
    """Schedule cancellation with Stripe; never claim success without its response."""
    if not SETTINGS.payments_enabled:
        raise RuntimeError("Payments are not configured.")
    secret = os.getenv("STRIPE_SECRET_KEY", "")
    if not secret or not subscription_id:
        raise RuntimeError("No active Stripe subscription is connected to this account.")
    import stripe

    stripe.api_key = secret
    subscription = stripe.Subscription.modify(subscription_id, cancel_at_period_end=True)
    return {
        "id": str(subscription.get("id") or subscription_id),
        "status": str(subscription.get("status") or "active"),
        "cancel_at_period_end": bool(subscription.get("cancel_at_period_end")),
        "current_period_end": subscription.get("current_period_end"),
    }


def roster_capacity(plan: dict, current: int) -> dict:
    """Whether this workspace may add another fighter.

    The seat count is what a coach's plan actually sells, so it has to be
    enforced rather than described. Archived fighters do not count: someone who
    has left the gym should not hold a seat, and the fights they were in still
    resolve to their name.
    """
    limit = plan.get("roster_limit")
    if limit is None:
        return {"limit": None, "used": current, "remaining": None, "can_add": True}
    remaining = max(0, int(limit) - int(current))
    return {
        "limit": int(limit),
        "used": int(current),
        "remaining": remaining,
        "can_add": remaining > 0,
    }


def plans_for(audience: str) -> list[tuple[str, dict]]:
    """The plans worth showing this kind of workspace.

    A coach is never offered a one-fighter plan and an athlete is not sold
    seats they will not use, so each side sees a ladder that reads as a ladder
    rather than a catalogue with half of it irrelevant.
    """
    wanted = "coach" if str(audience).strip().lower() == "coach" else "athlete"
    return [(key, plan) for key, plan in PLANS.items() if plan.get("audience") == wanted]

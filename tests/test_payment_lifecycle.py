"""What each Stripe lifecycle event does to a plan (core.payments.subscription_change)."""

import unittest
from unittest import mock

from core import payments
from core.payments import subscription_change


def subscription_event(kind, status="active", cancel_at_period_end=False, price="price_athlete", end=1_800_000_000):
    return {"type": kind, "data": {"object": {
        "id": "sub_1", "customer": "cus_1", "status": status, "cancel_at_period_end": cancel_at_period_end,
        "items": {"data": [{"price": {"id": price}, "current_period_end": end}]}}}}


class SubscriptionChangeTests(unittest.TestCase):
    def setUp(self):
        prices = {key: dict(plan) for key, plan in payments.PLANS.items()}
        prices["athlete"]["stripe_price_id"] = "price_athlete"
        prices["athlete_pro"]["stripe_price_id"] = "price_pro"
        patcher = mock.patch.object(payments, "PLANS", prices)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_cancelled_subscription_ends_the_plan(self):
        change = subscription_change(subscription_event("customer.subscription.deleted", status="canceled"))
        self.assertEqual((change["plan"], change["status"], change["subscription_id"]), ("free", "cancelled", "sub_1"))

    def test_stripe_giving_up_on_the_card_ends_it_too(self):
        for status in ("unpaid", "canceled", "incomplete_expired"):
            self.assertEqual(subscription_change(subscription_event("customer.subscription.updated", status))["plan"], "free")

    def test_a_card_still_being_retried_keeps_the_plan(self):
        change = subscription_change(subscription_event("customer.subscription.updated", "past_due"))
        self.assertIsNotNone(change)
        self.assertEqual(change["status"], "past_due")
        self.assertNotEqual(change["plan"], "free")
        failed = subscription_change({"type": "invoice.payment_failed", "data": {"object": {
            "subscription": "sub_1", "customer": "cus_1", "lines": {"data": []}}}})
        self.assertEqual((failed["plan"], failed["status"]), (None, "past_due"))

    def test_cancel_at_period_end_keeps_access_until_then(self):
        change = subscription_change(subscription_event("customer.subscription.updated", cancel_at_period_end=True))
        self.assertEqual(change["status"], "cancel_at_period_end")
        self.assertEqual(change["plan"], "athlete")
        self.assertTrue(change["period_end"].startswith("2027-01-15"))

    def test_a_plan_change_follows_the_new_price(self):
        change = subscription_change(subscription_event("customer.subscription.updated", price="price_pro"))
        self.assertEqual(change["plan"], "athlete_pro")

    def test_an_unknown_price_changes_no_plan(self):
        self.assertIsNone(subscription_change(subscription_event("customer.subscription.updated", price="price_x"))["plan"])

    def test_a_renewal_extends_the_period(self):
        change = subscription_change({"type": "invoice.paid", "data": {"object": {
            "subscription": "sub_1", "customer": "cus_1", "lines": {"data": [{"period": {"end": 1_800_000_000}}]}}}})
        self.assertEqual((change["plan"], change["status"]), (None, "active"))
        self.assertTrue(change["period_end"].startswith("2027-01-15"))

    def test_only_a_full_refund_ends_the_plan(self):
        partial = {"type": "charge.refunded", "data": {"object": {"customer": "cus_1", "amount": 1000, "amount_refunded": 300}}}
        full = {"type": "charge.refunded", "data": {"object": {"customer": "cus_1", "amount": 1000, "amount_refunded": 1000}}}
        self.assertIsNone(subscription_change(partial))
        self.assertEqual(subscription_change(full)["plan"], "free")

    def test_a_chargeback_ends_the_plan(self):
        change = subscription_change({"type": "charge.dispute.created", "data": {"object": {"customer": "cus_1"}}})
        self.assertEqual((change["plan"], change["status"]), ("free", "disputed"))

    def test_other_events_change_nothing(self):
        self.assertIsNone(subscription_change({"type": "customer.created", "data": {"object": {}}}))


if __name__ == "__main__":
    unittest.main()

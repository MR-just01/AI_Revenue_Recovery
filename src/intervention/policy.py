from typing import Any


# --------------------------------------------------
# Phase 3 — Intervention Policy
# --------------------------------------------------

# This policy maps a detected recoverable case
# to a bounded recovery intervention.

#
# It only decides WHAT intervention is appropriate
# after the detection layer has identified a
# recoverable case.


# --------------------------------------------------
# Supported dataset values
# --------------------------------------------------

SUPPORTED_EVENTS = {
    "payment_failure",
    "checkout_abandonment",
    "subscription_failure",
    "overdue_invoice",
}


PAYMENT_RETRY_FAILURES = {
    "bank_timeout",
    "issuer_down",
    "network_error",
    "insufficient_funds",
    "3ds_failure",
     "otp_failure",
}



PAYMENT_METHOD_UPDATE_FAILURES = {
    "risk_declined",
    "card_expired",
    "mandate_limit_exceeded",
}


CHECKOUT_RECOVERY_FAILURES = {
    "payment_selection_dropoff",
    "cart_dropoff",
    "otp_dropoff",
    "address_dropoff",
    "3ds_failure",
    "otp_failure",
}


SUBSCRIPTION_FAILURES = {
    "insufficient_funds",
    "risk_declined",
    "card_expired",
    "mandate_limit_exceeded",
    "issuer_down",
    "bank_timeout",
    "network_error",
}


OVERDUE_INVOICE_FAILURES = {
    "overdue_1_30d",
    "overdue_31_60d",
    "overdue_60d_plus",
}


# --------------------------------------------------
# Main policy function
# --------------------------------------------------

def determine_intervention(
    *,
    risk: str,
    event: str,
    failure: str,
    amount: float | None = None,
    prior_failed_attempts: int | None = None,
) -> dict[str, Any]:
    """
    Determine the appropriate recovery intervention.

    Parameters
    ----------
    risk:
        Detection result. Expected values:
        "recoverable" or "not_recoverable".

    event:
        Revenue-risk event from the frozen dataset.

    failure:
        Specific failure reason from the frozen dataset.

    amount:
        Transaction/invoice amount. Currently retained
        for future policy decisions but not used to
        create unsupported monetary rules.

    prior_failed_attempts:
        Number of previous failed attempts. Currently
        retained for future bounded-execution logic.

    Returns
    -------
    dict
        Structured intervention decision.
    """

    # --------------------------------------------------
    # Rule 0 — Risk gate
    # --------------------------------------------------

    if risk != "recoverable":
        return {
            "risk": risk,
            "intervention": "no_action",
            "reason": "risk_below_recovery_threshold",
            "priority": "none",
            "stop_rule": "stop",
        }


    # --------------------------------------------------
    # Validate event
    # --------------------------------------------------

    if event not in SUPPORTED_EVENTS:
        raise ValueError(
            f"Unsupported event: {event}"
        )


    # --------------------------------------------------
    # Rule 1 — Payment failure
    # --------------------------------------------------

    if event == "payment_failure":

        if failure in PAYMENT_RETRY_FAILURES:
            return {
                "risk": risk,
                "intervention": "payment_retry",
                "reason": failure,
                "priority": "high",
                "stop_rule": "stop_if_payment_succeeds_or_retry_limit_reached",
            }

        if failure in PAYMENT_METHOD_UPDATE_FAILURES:
            return {
                "risk": risk,
                "intervention": "payment_method_update",
                "reason": failure,
                "priority": "high",
                "stop_rule": "stop_after_payment_method_update_attempt",
            }

        raise ValueError(
            f"Unsupported payment failure reason: {failure}"
        )


    # --------------------------------------------------
    # Rule 2 — Checkout abandonment
    # --------------------------------------------------

    if event == "checkout_abandonment":

        if failure in CHECKOUT_RECOVERY_FAILURES:
            return {
                "risk": risk,
                "intervention": "checkout_recovery",
                "reason": failure,
                "priority": "medium",
                "stop_rule": "stop_after_checkout_recovery_attempt",
            }

        raise ValueError(
            f"Unsupported checkout abandonment reason: {failure}"
        )


    # --------------------------------------------------
    # Rule 3 — Subscription failure
    # --------------------------------------------------

    if event == "subscription_failure":

        if failure in SUBSCRIPTION_FAILURES:
            return {
                "risk": risk,
                "intervention": "subscription_recovery",
                "reason": failure,
                "priority": "high",
                "stop_rule": "stop_after_subscription_recovery_attempt",
            }

        raise ValueError(
            f"Unsupported subscription failure reason: {failure}"
        )


    # --------------------------------------------------
    # Rule 4 — Overdue invoice
    # --------------------------------------------------

    if event == "overdue_invoice":

        if failure == "overdue_1_30d":
            return {
                "risk": risk,
                "intervention": "receivables_chaser",
                "reason": failure,
                "priority": "medium",
                "stop_rule": "stop_after_receivables_chaser_limit",
            }

        if failure == "overdue_31_60d":
            return {
                "risk": risk,
                "intervention": "receivables_chaser",
                "reason": failure,
                "priority": "high",
                "stop_rule": "stop_after_receivables_chaser_limit",
            }

        if failure == "overdue_60d_plus":
            return {
                "risk": risk,
                "intervention": "receivables_escalation",
                "reason": failure,
                "priority": "high",
                "stop_rule": "stop_after_escalation_attempt",
            }

        raise ValueError(
            f"Unsupported overdue invoice reason: {failure}"
        )


    # --------------------------------------------------
    # Safety fallback
    # --------------------------------------------------

    raise ValueError(
        f"No intervention rule exists for event={event}, "
        f"failure={failure}"
    )


# --------------------------------------------------
# Simple manual test
# --------------------------------------------------

if __name__ == "__main__":

    examples = [
        {
            "risk": "recoverable",
            "event": "payment_failure",
            "failure": "bank_timeout",
        },
        {
            "risk": "recoverable",
            "event": "payment_failure",
            "failure": "card_expired",
        },
        {
            "risk": "recoverable",
            "event": "checkout_abandonment",
            "failure": "otp_dropoff",
        },
        {
            "risk": "recoverable",
            "event": "subscription_failure",
            "failure": "insufficient_funds",
        },
        {
            "risk": "recoverable",
            "event": "overdue_invoice",
            "failure": "overdue_60d_plus",
        },
        {
            "risk": "not_recoverable",
            "event": "payment_failure",
            "failure": "bank_timeout",
        },
    ]

    print("=" * 60)
    print("PHASE 3 — INTERVENTION POLICY")
    print("=" * 60)

    for example in examples:

        result = determine_intervention(**example)

        print("\nInput:")
        print(example)

        print("Decision:")
        print(result)

    print("\nPolicy evaluation completed.")
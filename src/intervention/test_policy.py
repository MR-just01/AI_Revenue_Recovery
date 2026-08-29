"""
Phase 3 — Intervention Policy Tests
These tests validate the deterministic intervention policy against
the actual frozen Phase 1 dataset.
"""

from pathlib import Path
import sys

import pandas as pd

# --------------------------------------------------
# Make src importable when this file is run directly
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.intervention.policy import (  # noqa: E402
    CHECKOUT_RECOVERY_FAILURES,
    OVERDUE_INVOICE_FAILURES,
    PAYMENT_METHOD_UPDATE_FAILURES,
    PAYMENT_RETRY_FAILURES,
    SUBSCRIPTION_FAILURES,
    SUPPORTED_EVENTS,
    determine_intervention,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_FILE = ROOT / "data" / "revenue_recovery_full.csv"


# --------------------------------------------------
# Test helpers
# --------------------------------------------------

def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Frozen dataset not found: {DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


# --------------------------------------------------
# Test 1 — Every dataset event/failure combination
# --------------------------------------------------

def test_all_dataset_combinations(df: pd.DataFrame) -> None:
    """
    Every unique event + failure combination present in the
    frozen dataset must have a valid policy result.
    """

    combinations = (
        df[["event", "failure"]]
        .drop_duplicates()
        .sort_values(["event", "failure"])
    )

    for row in combinations.itertuples(index=False):

        result = determine_intervention(
            risk="recoverable",
            event=row.event,
            failure=row.failure,
        )

        required_keys = {
            "risk",
            "intervention",
            "reason",
            "priority",
            "stop_rule",
        }

        assert_true(
            isinstance(result, dict),
            f"Policy did not return a dict for {row.event} / {row.failure}",
        )

        assert_true(
            required_keys.issubset(result.keys()),
            f"Missing policy fields for {row.event} / {row.failure}",
        )

        assert_true(
            result["risk"] == "recoverable",
            f"Unexpected risk output for {row.event} / {row.failure}",
        )

        assert_true(
            result["intervention"] != "no_action",
            f"Recoverable case unexpectedly mapped to no_action: "
            f"{row.event} / {row.failure}",
        )


# --------------------------------------------------
# Test 2 — Risk gate
# --------------------------------------------------

def test_risk_gate(df: pd.DataFrame) -> None:
    """
    Any not_recoverable case must stop without an active
    recovery intervention.
    """

    combinations = (
        df[["event", "failure"]]
        .drop_duplicates()
    )

    for row in combinations.itertuples(index=False):

        result = determine_intervention(
            risk="not_recoverable",
            event=row.event,
            failure=row.failure,
        )

        assert_true(
            result["risk"] == "not_recoverable",
            "Risk gate changed the supplied risk.",
        )

        assert_true(
            result["intervention"] == "no_action",
            f"not_recoverable case received an intervention: "
            f"{row.event} / {row.failure}",
        )

        assert_true(
            result["priority"] == "none",
            "not_recoverable case did not receive priority=none.",
        )

        assert_true(
            result["stop_rule"] == "stop",
            "not_recoverable case does not stop immediately.",
        )


# --------------------------------------------------
# Test 3 — Payment failure mappings
# --------------------------------------------------

def test_payment_failure_rules() -> None:

    for failure in PAYMENT_RETRY_FAILURES:

        result = determine_intervention(
            risk="recoverable",
            event="payment_failure",
            failure=failure,
        )

        assert_true(
            result["intervention"] == "payment_retry",
            f"Expected payment_retry for {failure}, got "
            f"{result['intervention']}",
        )

    for failure in PAYMENT_METHOD_UPDATE_FAILURES:

        result = determine_intervention(
            risk="recoverable",
            event="payment_failure",
            failure=failure,
        )

        assert_true(
            result["intervention"] == "payment_method_update",
            f"Expected payment_method_update for {failure}, got "
            f"{result['intervention']}",
        )


# --------------------------------------------------
# Test 4 — Checkout mappings
# --------------------------------------------------

def test_checkout_rules() -> None:

    for failure in CHECKOUT_RECOVERY_FAILURES:

        result = determine_intervention(
            risk="recoverable",
            event="checkout_abandonment",
            failure=failure,
        )

        assert_true(
            result["intervention"] == "checkout_recovery",
            f"Expected checkout_recovery for {failure}, got "
            f"{result['intervention']}",
        )


# --------------------------------------------------
# Test 5 — Subscription mappings
# --------------------------------------------------

def test_subscription_rules() -> None:

    for failure in SUBSCRIPTION_FAILURES:

        result = determine_intervention(
            risk="recoverable",
            event="subscription_failure",
            failure=failure,
        )

        assert_true(
            result["intervention"] == "subscription_recovery",
            f"Expected subscription_recovery for {failure}, got "
            f"{result['intervention']}",
        )


# --------------------------------------------------
# Test 6 — Overdue invoice mappings
# --------------------------------------------------

def test_overdue_invoice_rules() -> None:

    expected = {
        "overdue_1_30d": (
            "receivables_chaser",
            "medium",
        ),
        "overdue_31_60d": (
            "receivables_chaser",
            "high",
        ),
        "overdue_60d_plus": (
            "receivables_escalation",
            "high",
        ),
    }

    for failure in OVERDUE_INVOICE_FAILURES:

        result = determine_intervention(
            risk="recoverable",
            event="overdue_invoice",
            failure=failure,
        )

        expected_intervention, expected_priority = expected[failure]

        assert_true(
            result["intervention"] == expected_intervention,
            f"Unexpected intervention for {failure}: "
            f"{result['intervention']}",
        )

        assert_true(
            result["priority"] == expected_priority,
            f"Unexpected priority for {failure}: "
            f"{result['priority']}",
        )


# --------------------------------------------------
# Test 7 — All supported events are represented
# --------------------------------------------------

def test_supported_events(df: pd.DataFrame) -> None:

    observed_events = set(df["event"].unique())

    assert_true(
        observed_events == SUPPORTED_EVENTS,
        "Frozen dataset event set does not match policy-supported events. "
        f"Observed={observed_events}, Supported={SUPPORTED_EVENTS}",
    )


# --------------------------------------------------
# Test 8 — Invalid event is rejected
# --------------------------------------------------

def test_invalid_event() -> None:

    try:
        determine_intervention(
            risk="recoverable",
            event="unknown_event",
            failure="unknown_failure",
        )
    except ValueError:
        return

    raise AssertionError(
        "Unsupported event did not raise ValueError."
    )


# --------------------------------------------------
# Test 9 — Unknown failure for a valid event is rejected
# --------------------------------------------------

def test_invalid_failure() -> None:

    try:
        determine_intervention(
            risk="recoverable",
            event="payment_failure",
            failure="unknown_failure",
        )
    except ValueError:
        return

    raise AssertionError(
        "Unsupported failure reason did not raise ValueError."
    )


# --------------------------------------------------
# Test 10 — Required policy outputs are non-empty
# --------------------------------------------------

def test_outputs_are_complete(df: pd.DataFrame) -> None:

    recoverable_rows = df[df["risk"] == "recoverable"]

    for row in recoverable_rows.itertuples(index=False):

        result = determine_intervention(
            risk=row.risk,
            event=row.event,
            failure=row.failure,
            amount=row.amount,
            prior_failed_attempts=row.prior_failed_attempts,
        )

        for key in (
            "intervention",
            "reason",
            "priority",
            "stop_rule",
        ):
            value = result.get(key)

            assert_true(
                value is not None and str(value).strip() != "",
                f"Missing/empty {key} for record {row.record_id}",
            )


# --------------------------------------------------
# Main test runner
# --------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("PHASE 3 — INTERVENTION POLICY TESTS")
    print("=" * 60)

    df = load_dataset()

    tests = [
        ("all_dataset_combinations", lambda: test_all_dataset_combinations(df)),
        ("risk_gate", lambda: test_risk_gate(df)),
        ("payment_failure_rules", test_payment_failure_rules),
        ("checkout_rules", test_checkout_rules),
        ("subscription_rules", test_subscription_rules),
        ("overdue_invoice_rules", test_overdue_invoice_rules),
        ("supported_events", lambda: test_supported_events(df)),
        ("invalid_event", test_invalid_event),
        ("invalid_failure", test_invalid_failure),
        ("outputs_are_complete", lambda: test_outputs_are_complete(df)),
    ]

    passed = 0

    for name, test in tests:

        try:
            test()
            print(f"✓ {name}")
            passed += 1

        except Exception as exc:
            print(f"✗ {name}")
            print(f"  {exc}")
            raise

    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{len(tests)} TESTS PASSED")
    print("INTERVENTION POLICY VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

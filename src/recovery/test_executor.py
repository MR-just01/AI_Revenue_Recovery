"""
Phase 4 — Recovery Executor Tests

Run from repository root:

    python src\recovery\test_executor.py

These tests verify:
- no_action behavior
- escalation behavior
- successful recovery
- bounded retries
- customer opt-out
- invalid inputs
- recovered amount never exceeds the original amount
"""

from pathlib import Path
import random
import sys


# --------------------------------------------------
# Make src importable when run directly
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.recovery.executor import (  # noqa: E402
    MAX_ATTEMPTS_BY_INTERVENTION,
    execute_recovery,
)


# --------------------------------------------------
# Test helper
# --------------------------------------------------

def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------
# Test 1 — no_action
# --------------------------------------------------

def test_no_action() -> None:
    result = execute_recovery(
        intervention="no_action",
        amount=5000.0,
        success_probability=1.0,
        rng=random.Random(42),
    )

    assert_true(
        result["attempt"] == "0/0",
        f"Expected 0/0, got {result['attempt']}",
    )

    assert_true(
        result["result"] == "not_attempted",
        f"Expected not_attempted, got {result['result']}",
    )

    assert_true(
        result["recovered"] == 0.0,
        f"Expected 0 recovered, got {result['recovered']}",
    )

    assert_true(
        result["status"] == "not_recovered",
        f"Unexpected status: {result['status']}",
    )


# --------------------------------------------------
# Test 2 — escalation
# --------------------------------------------------

def test_escalation() -> None:
    result = execute_recovery(
        intervention="receivables_escalation",
        amount=50000.0,
        success_probability=1.0,
        rng=random.Random(42),
    )

    assert_true(
        result["attempt"] == "1/1",
        f"Expected 1/1, got {result['attempt']}",
    )

    assert_true(
        result["status"] == "escalated",
        f"Expected escalated, got {result['status']}",
    )

    assert_true(
        result["recovered"] == 0.0,
        "Escalation should not directly recover money.",
    )

    assert_true(
        result["stop_reason"] == "escalated_unresolved",
        f"Unexpected stop reason: {result['stop_reason']}",
    )


# --------------------------------------------------
# Test 3 — guaranteed success
# --------------------------------------------------

def test_success_on_first_attempt() -> None:
    result = execute_recovery(
        intervention="payment_retry",
        amount=5000.0,
        success_probability=1.0,
        opt_out_probability=0.0,
        rng=random.Random(42),
    )

    assert_true(
        result["attempt"] == "1/3",
        f"Expected 1/3, got {result['attempt']}",
    )

    assert_true(
        result["result"] == "success",
        f"Expected success, got {result['result']}",
    )

    assert_true(
        result["recovered"] == 5000.0,
        f"Expected 5000 recovered, got {result['recovered']}",
    )

    assert_true(
        result["status"] == "recovered",
        f"Expected recovered status, got {result['status']}",
    )

    assert_true(
        result["stop_reason"] == "payment_success",
        f"Unexpected stop reason: {result['stop_reason']}",
    )


# --------------------------------------------------
# Test 4 — bounded failure
# --------------------------------------------------

def test_max_attempts() -> None:
    result = execute_recovery(
        intervention="payment_retry",
        amount=5000.0,
        success_probability=0.0,
        opt_out_probability=0.0,
        rng=random.Random(42),
    )

    expected_attempt = (
        f"{MAX_ATTEMPTS_BY_INTERVENTION['payment_retry']}/"
        f"{MAX_ATTEMPTS_BY_INTERVENTION['payment_retry']}"
    )

    assert_true(
        result["attempt"] == expected_attempt,
        f"Expected {expected_attempt}, got {result['attempt']}",
    )

    assert_true(
        result["result"] == "failure",
        f"Expected failure, got {result['result']}",
    )

    assert_true(
        result["recovered"] == 0.0,
        "Failed recovery should recover zero amount.",
    )

    assert_true(
        result["status"] == "failed",
        f"Expected failed status, got {result['status']}",
    )

    assert_true(
        result["stop_reason"] == "max_retries_reached",
        f"Unexpected stop reason: {result['stop_reason']}",
    )


# --------------------------------------------------
# Test 5 — customer opt-out
# --------------------------------------------------

def test_customer_opt_out() -> None:
    result = execute_recovery(
        intervention="payment_retry",
        amount=5000.0,
        success_probability=1.0,
        opt_out_probability=1.0,
        rng=random.Random(42),
    )

    assert_true(
        result["attempt"] == "1/3",
        f"Expected first attempt, got {result['attempt']}",
    )

    assert_true(
        result["result"] == "failure",
        f"Expected failure, got {result['result']}",
    )

    assert_true(
        result["recovered"] == 0.0,
        "Opt-out must not recover money.",
    )

    assert_true(
        result["status"] == "stopped",
        f"Expected stopped status, got {result['status']}",
    )

    assert_true(
        result["stop_reason"] == "customer_opted_out",
        f"Unexpected stop reason: {result['stop_reason']}",
    )


# --------------------------------------------------
# Test 6 — amount cannot be negative
# --------------------------------------------------

def test_negative_amount_rejected() -> None:
    try:
        execute_recovery(
            intervention="payment_retry",
            amount=-100.0,
            success_probability=0.5,
        )
    except ValueError:
        return

    raise AssertionError(
        "Negative amount was not rejected."
    )


# --------------------------------------------------
# Test 7 — invalid probability rejected
# --------------------------------------------------

def test_invalid_probability_rejected() -> None:
    try:
        execute_recovery(
            intervention="payment_retry",
            amount=1000.0,
            success_probability=1.5,
        )
    except ValueError:
        return

    raise AssertionError(
        "Invalid success probability was not rejected."
    )


# --------------------------------------------------
# Test 8 — invalid intervention rejected
# --------------------------------------------------

def test_invalid_intervention_rejected() -> None:
    try:
        execute_recovery(
            intervention="unknown_intervention",
            amount=1000.0,
            success_probability=0.5,
        )
    except ValueError:
        return

    raise AssertionError(
        "Unsupported intervention was not rejected."
    )


# --------------------------------------------------
# Test 9 — recovered amount cannot exceed amount
# --------------------------------------------------

def test_recovered_amount_never_exceeds_amount() -> None:
    interventions = [
        "payment_retry",
        "payment_method_update",
        "checkout_recovery",
        "subscription_recovery",
        "receivables_chaser",
    ]

    for intervention in interventions:

        result = execute_recovery(
            intervention=intervention,
            amount=2500.0,
            success_probability=1.0,
            opt_out_probability=0.0,
            rng=random.Random(42),
        )

        assert_true(
            result["recovered"] <= 2500.0,
            f"{intervention} recovered more than the event amount.",
        )


# --------------------------------------------------
# Test 10 — all bounded interventions have limits
# --------------------------------------------------

def test_all_interventions_have_bounds() -> None:
    expected_interventions = {
        "payment_retry",
        "payment_method_update",
        "checkout_recovery",
        "subscription_recovery",
        "receivables_chaser",
        "receivables_escalation",
    }

    configured_interventions = set(
        MAX_ATTEMPTS_BY_INTERVENTION.keys()
    )

    assert_true(
        expected_interventions == configured_interventions,
        "Intervention bounds are incomplete.",
    )

    for intervention, limit in MAX_ATTEMPTS_BY_INTERVENTION.items():

        assert_true(
            isinstance(limit, int),
            f"{intervention} has a non-integer attempt limit.",
        )

        assert_true(
            limit >= 1,
            f"{intervention} has an invalid attempt limit: {limit}",
        )


# --------------------------------------------------
# Main runner
# --------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("PHASE 4 — RECOVERY EXECUTOR TESTS")
    print("=" * 60)

    tests = [
        ("no_action", test_no_action),
        ("escalation", test_escalation),
        ("success_on_first_attempt", test_success_on_first_attempt),
        ("max_attempts", test_max_attempts),
        ("customer_opt_out", test_customer_opt_out),
        ("negative_amount_rejected", test_negative_amount_rejected),
        ("invalid_probability_rejected", test_invalid_probability_rejected),
        ("invalid_intervention_rejected", test_invalid_intervention_rejected),
        (
            "recovered_amount_never_exceeds_amount",
            test_recovered_amount_never_exceeds_amount,
        ),
        (
            "all_interventions_have_bounds",
            test_all_interventions_have_bounds,
        ),
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
    print("RECOVERY EXECUTOR VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
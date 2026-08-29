"""
Phase 4 — Bounded Recovery Executor

The executor:
- accepts an intervention chosen by the Phase 3 policy
- enforces a maximum number of attempts
- stops on success, opt-out, escalation, or attempt exhaustion
- returns a structured execution/outcome record

It does NOT:
- train the ML model
- select a threshold
- choose a new intervention

For the synthetic buildathon environment, a success probability is
supplied by the simulation/test environment. The executor itself does
not read p_recover_true from the dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Optional


# --------------------------------------------------
# Bounded intervention configuration
# --------------------------------------------------

MAX_ATTEMPTS_BY_INTERVENTION = {
    "payment_retry": 3,
    "payment_method_update": 1,
    "checkout_recovery": 2,
    "subscription_recovery": 3,
    "receivables_chaser": 3,
    "receivables_escalation": 1,
}

NO_ACTION_INTERVENTIONS = {
    "no_action",
}

ESCALATION_INTERVENTIONS = {
    "receivables_escalation",
}


# --------------------------------------------------
# Execution result
# --------------------------------------------------

@dataclass(frozen=True)
class ExecutionResult:
    intervention: str
    attempt: str
    result: str
    recovered: float
    status: str
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------
# Input validation
# --------------------------------------------------

def _validate_inputs(
    intervention: str,
    amount: float,
    success_probability: float,
    opt_out_probability: float,
) -> None:

    supported_interventions = (
        set(MAX_ATTEMPTS_BY_INTERVENTION)
        | NO_ACTION_INTERVENTIONS
        | ESCALATION_INTERVENTIONS
    )

    if intervention not in supported_interventions:
        raise ValueError(
            f"Unsupported intervention: {intervention}"
        )

    if amount < 0:
        raise ValueError(
            "amount cannot be negative."
        )

    if not 0.0 <= success_probability <= 1.0:
        raise ValueError(
            "success_probability must be between 0 and 1."
        )

    if not 0.0 <= opt_out_probability <= 1.0:
        raise ValueError(
            "opt_out_probability must be between 0 and 1."
        )


# --------------------------------------------------
# Recovery executor
# --------------------------------------------------

def execute_recovery(
    *,
    intervention: str,
    amount: float,
    success_probability: float,
    opt_out_probability: float = 0.06,
    decay: float = 0.72,
    rng: Optional[random.Random] = None,
) -> dict[str, Any]:
    """
    Execute a bounded recovery workflow.

    Parameters
    ----------
    intervention:
        Intervention selected by Phase 3 policy.py.

    amount:
        Revenue associated with the case.

    success_probability:
        Probability that an attempted recovery succeeds in the
        simulation environment.

    opt_out_probability:
        Probability that the customer opts out during an attempt.

    decay:
        Reduction in success probability for each subsequent attempt.

    rng:
        Optional seeded random generator for reproducible tests.

    Returns
    -------
    dict
        Structured execution result.
    """

    _validate_inputs(
        intervention=intervention,
        amount=amount,
        success_probability=success_probability,
        opt_out_probability=opt_out_probability,
    )

    if not 0.0 <= decay <= 1.0:
        raise ValueError(
            "decay must be between 0 and 1."
        )

    if rng is None:
        rng = random.Random()

    # --------------------------------------------------
    # No action
    # --------------------------------------------------

    if intervention in NO_ACTION_INTERVENTIONS:

        return ExecutionResult(
            intervention=intervention,
            attempt="0/0",
            result="not_attempted",
            recovered=0.0,
            status="not_recovered",
            stop_reason="no_action_required",
        ).to_dict()

    # --------------------------------------------------
    # Escalation
    # --------------------------------------------------

    if intervention in ESCALATION_INTERVENTIONS:

        return ExecutionResult(
            intervention=intervention,
            attempt="1/1",
            result="failure",
            recovered=0.0,
            status="escalated",
            stop_reason="escalated_unresolved",
        ).to_dict()

    # --------------------------------------------------
    # Bounded automated recovery
    # --------------------------------------------------

    max_attempts = MAX_ATTEMPTS_BY_INTERVENTION[
        intervention
    ]

    for attempt_number in range(
        1,
        max_attempts + 1,
    ):

        # Customer opt-out is an immediate stop condition.
        if rng.random() < opt_out_probability:

            return ExecutionResult(
                intervention=intervention,
                attempt=(
                    f"{attempt_number}/{max_attempts}"
                ),
                result="failure",
                recovered=0.0,
                status="stopped",
                stop_reason="customer_opted_out",
            ).to_dict()

        # Later attempts have lower probability of success.
        attempt_probability = min(
            max(
                success_probability
                * (decay ** (attempt_number - 1)),
                0.02,
            ),
            0.95,
        )

        if rng.random() < attempt_probability:

            if intervention in {
                "payment_retry",
                "payment_method_update",
            }:
                stop_reason = "payment_success"
            else:
                stop_reason = "customer_paid"

            return ExecutionResult(
                intervention=intervention,
                attempt=(
                    f"{attempt_number}/{max_attempts}"
                ),
                result="success",
                recovered=round(amount, 2),
                status="recovered",
                stop_reason=stop_reason,
            ).to_dict()

    # --------------------------------------------------
    # Attempts exhausted
    # --------------------------------------------------

    return ExecutionResult(
        intervention=intervention,
        attempt=(
            f"{max_attempts}/{max_attempts}"
        ),
        result="failure",
        recovered=0.0,
        status="failed",
        stop_reason="max_retries_reached",
    ).to_dict()


# --------------------------------------------------
# Manual smoke test
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("PHASE 4 — BOUNDED RECOVERY EXECUTOR")
    print("=" * 60)

    # Fixed seed makes this smoke test reproducible.
    rng = random.Random(42)

    examples = [
        {
            "intervention": "payment_retry",
            "amount": 5000.0,
            "success_probability": 0.80,
        },
        {
            "intervention": "checkout_recovery",
            "amount": 2500.0,
            "success_probability": 0.30,
        },
        {
            "intervention": "receivables_escalation",
            "amount": 50000.0,
            "success_probability": 0.10,
        },
        {
            "intervention": "no_action",
            "amount": 1000.0,
            "success_probability": 0.90,
        },
    ]

    for example in examples:

        result = execute_recovery(
            **example,
            rng=rng,
        )

        print("\nInput:")
        print(example)

        print("Execution result:")
        print(result)

    print("\nExecutor smoke test completed.")
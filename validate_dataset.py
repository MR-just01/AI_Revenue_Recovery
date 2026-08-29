"""
Phase 1 — AI Revenue Recovery dataset validation.

This validator is intentionally strict for the frozen Phase 1 baseline.
It checks schema, split integrity, missing-value semantics, categorical values,
numeric sanity, recovery consistency, leakage exclusions, and file hashes.
"""

from pathlib import Path
import hashlib
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

FULL_FILE = DATA_DIR / "revenue_recovery_full.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

EXPECTED_ROWS = {
    "revenue_recovery_full.csv": 600,
    "train.csv": 476,
    "test.csv": 124,
}

EXPECTED_SHA256 = {
    "revenue_recovery_full.csv": "1fed7e226f6a910b2ed3e0abfa3647230e2fbceef44dd8094907ff8ac173e5b2",
    "train.csv": "83f381a5ba2520eb2acedf536cd21daf0e1464749ca4677eb66a10c4fb3e17fb",
    "test.csv": "39b86e52142a20a9a488cf23c7d0fbd5181dce715afd332d7fba5bcaed876f4f",
}

EXPECTED_COLUMNS = [
    "record_id",
    "customer_id",
    "merchant_id",
    "created_at",
    "event",
    "amount",
    "failure",
    "root_cause",
    "risk",
    "agent_decision",
    "action",
    "attempt",
    "result",
    "recovered",
    "status",
    "stop_reason",
    "payment_method",
    "prior_failed_attempts",
    "customer_risk_score",
    "time_since_last_purchase_days",
    "historical_ltv_inr",
    "days_overdue",
    "p_recover_true",
]

ALLOWED_VALUES = {
    "event": {
        "payment_failure",
        "checkout_abandonment",
        "subscription_failure",
        "overdue_invoice",
    },
    "failure": {
        "payment_selection_dropoff",
        "overdue_31_60d",
        "address_dropoff",
        "overdue_1_30d",
        "insufficient_funds",
        "issuer_down",
        "otp_dropoff",
        "mandate_limit_exceeded",
        "otp_failure",
        "3ds_failure",
        "network_error",
        "risk_declined",
        "card_expired",
        "cart_dropoff",
        "bank_timeout",
        "overdue_60d_plus",
    },
    "root_cause": {
        "customer_hesitation",
        "hard_decline",
        "soft_decline",
        "technical_issue",
    },
    "risk": {"recoverable", "not_recoverable"},
    "agent_decision": {
        "send_reminder",
        "write_off",
        "soft_reminder_low_priority",
        "retry_alternate_method",
        "retry_later",
        "retry_prompt",
        "retry_now",
        "send_payment_reminder",
        "escalate_to_collections",
        "offer_alternate_payment",
    },
    "action": {
        "send_whatsapp_reminder",
        "no_action",
        "send_email_reminder",
        "retry_payment_alt_method",
        "retry_payment",
        "send_retry_link",
        "send_invoice_reminder",
        "escalate_to_human",
        "send_payment_link",
        "send_sms_reminder",
    },
    "result": {"success", "not_attempted", "failure"},
    "status": {"recovered", "not_recovered", "failed", "stopped", "escalated"},
    "stop_reason": {
        "customer_paid",
        "predicted_not_recoverable_skipped",
        "max_retries_reached",
        "payment_success",
        "customer_opted_out",
        "escalated_unresolved",
    },
    "payment_method": {
        "netbanking",
        "invoice",
        "upi",
        "card",
        "emi",
        "wallet",
    },
}

NUMERIC_RANGES = {
    "amount": (50, 128450),
    "prior_failed_attempts": (0, 4),
    "customer_risk_score": (0.011, 0.844),
    "time_since_last_purchase_days": (0, 399),
    "historical_ltv_inr": (53.13, 76414.76),
    "p_recover_true": (0.03, 0.97),
    "recovered": (0, 128450),
}

ML_TARGETS = {"risk", "root_cause"}

DOWNSTREAM_FIELDS = {
    "agent_decision",
    "action",
    "attempt",
    "result",
    "recovered",
    "status",
    "stop_reason",
}

FORBIDDEN_MODEL_FEATURES = DOWNSTREAM_FIELDS | ML_TARGETS | {"p_recover_true"}

ATTEMPT_PATTERN = re.compile(r"^(0/0|[1-9][0-9]*/[1-9][0-9]*)$")


class ValidationError(Exception):
    pass


def fail(message: str):
    raise ValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_files_exist():
    for path in (FULL_FILE, TRAIN_FILE, TEST_FILE):
        if not path.exists():
            fail(f"Missing required file: {path}")


def check_hashes():
    for path in (FULL_FILE, TRAIN_FILE, TEST_FILE):
        actual = sha256(path)
        expected = EXPECTED_SHA256[path.name]
        if actual != expected:
            fail(
                f"Frozen file hash mismatch for {path.name}. "
                f"Expected {expected}, got {actual}. "
                "If the dataset was intentionally changed, create a new dataset version."
            )


def load(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        fail(f"Could not read {path.name}: {exc}")


def check_shape_and_schema(frames):
    for name, df in frames.items():
        expected_rows = EXPECTED_ROWS[name]
        if len(df) != expected_rows:
            fail(f"{name}: expected {expected_rows} rows, found {len(df)}.")
        if len(df.columns) != len(EXPECTED_COLUMNS):
            fail(f"{name}: expected {len(EXPECTED_COLUMNS)} columns, found {len(df.columns)}.")
        if list(df.columns) != EXPECTED_COLUMNS:
            fail(f"{name}: column order/schema does not match schema.md.")


def check_missing_values(full: pd.DataFrame):
    required = [c for c in EXPECTED_COLUMNS if c != "days_overdue"]
    missing = full[required].isna().sum()
    bad = missing[missing > 0]
    if not bad.empty:
        fail(f"Unexpected missing values: {bad.to_dict()}")

    invoice_missing = full.loc[
        full["event"].eq("overdue_invoice"), "days_overdue"
    ].isna().sum()
    non_invoice_present = full.loc[
        ~full["event"].eq("overdue_invoice"), "days_overdue"
    ].notna().sum()

    if invoice_missing:
        fail(f"overdue_invoice rows missing days_overdue: {invoice_missing}")
    if non_invoice_present:
        fail(
            "Non-overdue-invoice rows unexpectedly contain days_overdue: "
            f"{non_invoice_present}"
        )


def check_ids(full: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame):
    if not full["record_id"].is_unique:
        fail("record_id is not unique in the full dataset.")

    train_ids = set(train["record_id"])
    test_ids = set(test["record_id"])
    full_ids = set(full["record_id"])

    overlap = train_ids & test_ids
    if overlap:
        fail(f"Train/test record_id overlap found: {sorted(overlap)[:10]}")

    if train_ids | test_ids != full_ids:
        fail("Train/test record IDs do not exactly cover the full dataset.")

    if len(train_ids) != len(train) or len(test_ids) != len(test):
        fail("Duplicate record_id found inside train or test.")


def check_duplicates(full: pd.DataFrame):
    duplicate_rows = full.duplicated().sum()
    if duplicate_rows:
        fail(f"Found {duplicate_rows} duplicate rows in full dataset.")


def check_categories(full: pd.DataFrame):
    for column, allowed in ALLOWED_VALUES.items():
        observed = set(full[column].dropna().astype(str).unique())
        invalid = observed - allowed
        if invalid:
            fail(f"{column}: unexpected values: {sorted(invalid)}")


def check_numeric_ranges(full: pd.DataFrame):
    for column, (lower, upper) in NUMERIC_RANGES.items():
        values = pd.to_numeric(full[column], errors="coerce")
        if values.isna().any():
            fail(f"{column}: contains non-numeric or missing values.")
        actual_min = values.min()
        actual_max = values.max()
        if actual_min < lower or actual_max > upper:
            fail(
                f"{column}: observed range {actual_min}–{actual_max} "
                f"falls outside frozen range {lower}–{upper}."
            )


def check_recovery_consistency(full: pd.DataFrame):
    if (full["recovered"] < 0).any():
        fail("recovered contains negative values.")

    if (full["recovered"] > full["amount"]).any():
        fail("recovered exceeds amount in at least one record.")

    positive_recovery = full["recovered"] > 0
    recovered_status = full["status"].eq("recovered")

    if not (positive_recovery == recovered_status).all():
        fail("recovered > 0 and status == recovered are inconsistent.")

    result_success = full["result"].eq("success")
    if not (result_success == positive_recovery).all():
        fail("result == success and recovered > 0 are inconsistent.")

    if not full["attempt"].astype(str).map(ATTEMPT_PATTERN.fullmatch).all():
        fail("attempt contains a value outside the expected N/M format.")

    # Current baseline: non-attempted records are exactly 0/0 + not_attempted.
    non_attempted = full["attempt"].eq("0/0")
    result_not_attempted = full["result"].eq("not_attempted")
    if not (non_attempted == result_not_attempted).all():
        fail("0/0 attempt and not_attempted result are inconsistent.")

    # Current baseline: a recovered case must have a successful execution.
    if not (full.loc[recovered_status, "result"].eq("success").all()):
        fail("Recovered status contains a non-success result.")

    # Current baseline: unrecovered/stopped/failed/escalated cases have zero recovery.
    if not (full.loc[~recovered_status, "recovered"].eq(0).all()):
        fail("Non-recovered statuses contain a positive recovered amount.")


def check_leakage_contract():
    baseline_detection_features = {
        "event",
        "amount",
        "failure",
        "payment_method",
        "prior_failed_attempts",
        "customer_risk_score",
        "time_since_last_purchase_days",
        "historical_ltv_inr",
        "days_overdue",
    }

    if baseline_detection_features & FORBIDDEN_MODEL_FEATURES:
        fail("Detection feature contract accidentally includes a forbidden field.")

    if "p_recover_true" not in FORBIDDEN_MODEL_FEATURES:
        fail("p_recover_true was not excluded from model features.")

    # This check documents the intended feature boundary in executable form.
    expected = {
        "event",
        "amount",
        "failure",
        "payment_method",
        "prior_failed_attempts",
        "customer_risk_score",
        "time_since_last_purchase_days",
        "historical_ltv_inr",
        "days_overdue",
    }
    if baseline_detection_features != expected:
        fail("Detection feature contract does not match the Phase 1 schema.")


def check_datetime(full: pd.DataFrame):
    parsed = pd.to_datetime(full["created_at"], errors="coerce")
    if parsed.isna().any():
        fail("created_at contains invalid datetime values.")


def main():
    print("AI Revenue Recovery — Phase 1 Dataset Validation")
    print("=" * 54)

    try:
        check_files_exist()

        full = load(FULL_FILE)
        train = load(TRAIN_FILE)
        test = load(TEST_FILE)

        frames = {
            "revenue_recovery_full.csv": full,
            "train.csv": train,
            "test.csv": test,
        }

        check_shape_and_schema(frames)
        check_ids(full, train, test)
        check_duplicates(full)
        check_missing_values(full)
        check_categories(full)
        check_numeric_ranges(full)
        check_datetime(full)
        check_recovery_consistency(full)
        check_leakage_contract()
        check_hashes()

    except ValidationError as exc:
        print(f"\n✗ VALIDATION FAILED\n  {exc}")
        sys.exit(1)

    print("\n✓ Schema valid — 23 columns")
    print("✓ Full dataset valid — 600 records")
    print("✓ Train/test sizes valid — 476 / 124")
    print("✓ Train/test record IDs have no overlap")
    print("✓ Train + test exactly cover the full dataset")
    print("✓ No duplicate full-dataset rows")
    print("✓ Required fields have no unexpected missing values")
    print("✓ days_overdue follows event-specific missingness")
    print("✓ Categorical values are valid")
    print("✓ Numeric ranges are valid")
    print("✓ created_at contains valid datetimes")
    print("✓ Recovery amounts and statuses are consistent")
    print("✓ Execution results and attempt format are consistent")
    print("✓ ML leakage exclusions are defined")
    print("✓ Frozen SHA-256 file hashes match")
    print("\nDATASET VALIDATION PASSED")
    print("Phase 1 dataset is ready to freeze.")


if __name__ == "__main__":
    main()

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "detection_logistic_regression.joblib"
METRICS_FILE = MODEL_DIR / "detection_validation_metrics.json"
TEST_METRICS_FILE = MODEL_DIR / "detection_test_metrics.json"


# --------------------------------------------------
# Detection contract
# --------------------------------------------------

FEATURES = [
    "event",
    "amount",
    "failure",
    "payment_method",
    "prior_failed_attempts",
    "customer_risk_score",
    "time_since_last_purchase_days",
    "historical_ltv_inr",
    "days_overdue",
]

TARGET = "risk"


# --------------------------------------------------
# Load frozen model
# --------------------------------------------------

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Frozen model not found: {MODEL_FILE}"
    )

pipeline = joblib.load(MODEL_FILE)


# --------------------------------------------------
# Load frozen threshold
# --------------------------------------------------

if not METRICS_FILE.exists():
    raise FileNotFoundError(
        f"Validation metrics not found: {METRICS_FILE}"
    )

with METRICS_FILE.open(
    "r",
    encoding="utf-8",
) as f:
    validation_metrics = json.load(f)


threshold = validation_metrics["threshold"]


# --------------------------------------------------
# Load test data
# --------------------------------------------------

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_FILE}"
    )

df = pd.read_csv(TEST_FILE)


# --------------------------------------------------
# Validate test schema
# --------------------------------------------------

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Test dataset is missing required columns: "
        f"{missing_columns}"
    )


# --------------------------------------------------
# Prepare test data
# --------------------------------------------------

X_test = df[FEATURES]

y_test = (
    df[TARGET] == "recoverable"
).astype(int)


# --------------------------------------------------
# Generate test probabilities
# --------------------------------------------------

# IMPORTANT:
# No training happens here.
# No threshold selection happens here.
#
# The frozen model only produces probabilities.

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# Apply frozen validation threshold
# --------------------------------------------------

predictions = (
    probabilities >= threshold
).astype(int)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
).ravel()


# --------------------------------------------------
# Test metrics
# --------------------------------------------------

metrics = {
    "model": "logistic_regression",
    "target": TARGET,
    "positive_class": "recoverable",
    "negative_class": "not_recoverable",

    # This threshold was selected using validation data.
    "frozen_threshold": threshold,

    "test_rows": len(X_test),

    "true_positives": int(tp),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),

    "recall_recoverable": float(
        recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
    ),

    "precision_recoverable": float(
        precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
    ),

    "f1": float(
        f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
    ),

    "pr_auc": float(
        average_precision_score(
            y_test,
            probabilities,
        )
    ),

    "roc_auc": float(
        roc_auc_score(
            y_test,
            probabilities,
        )
    ),

    "brier_score": float(
        brier_score_loss(
            y_test,
            probabilities,
        )
    ),

    "balanced_accuracy": float(
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    ),
}


# --------------------------------------------------
# Save test metrics
# --------------------------------------------------

with TEST_METRICS_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics,
        f,
        indent=2,
    )


# --------------------------------------------------
# Output
# --------------------------------------------------

print("=" * 60)
print("PHASE 2 — TEST EVALUATION")
print("=" * 60)

print(
    f"Test rows:       {len(X_test)}"
)

print(
    f"Frozen threshold: {threshold:.2f}"
)

print(
    "Threshold source: validation set"
)


print("\nConfusion Matrix")

print(
    f"TN: {tn}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print(
    f"TP: {tp}"
)


print("\nPrimary Metrics")

print(
    f"Recall:    "
    f"{metrics['recall_recoverable']:.4f}"
)

print(
    f"Precision: "
    f"{metrics['precision_recoverable']:.4f}"
)


print("\nSupporting Metrics")

print(
    f"PR-AUC:          "
    f"{metrics['pr_auc']:.4f}"
)

print(
    f"F1:              "
    f"{metrics['f1']:.4f}"
)

print(
    f"ROC-AUC:         "
    f"{metrics['roc_auc']:.4f}"
)

print(
    f"Brier Score:     "
    f"{metrics['brier_score']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{metrics['balanced_accuracy']:.4f}"
)


print("\nArtifacts")

print(
    f"Model:            "
    f"{MODEL_FILE}"
)

print(
    f"Validation metrics:"
    f" {METRICS_FILE}"
)

print(
    f"Test metrics:     "
    f"{TEST_METRICS_FILE}"
)

print("\nTest evaluation completed.")
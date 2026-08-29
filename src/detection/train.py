from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.csv"
MODEL_FILE = MODEL_DIR / "detection_logistic_regression.joblib"
METRICS_FILE = MODEL_DIR / "detection_validation_metrics.json"
THRESHOLD_FILE = MODEL_DIR / "threshold_analysis.csv"


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

CATEGORICAL_FEATURES = [
    "event",
    "failure",
    "payment_method",
]

NUMERIC_FEATURES = [
    "amount",
    "prior_failed_attempts",
    "customer_risk_score",
    "time_since_last_purchase_days",
    "historical_ltv_inr",
    "days_overdue",
]


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(TRAIN_FILE)

X = df[FEATURES]
y = (df[TARGET] == "recoverable").astype(int)


# --------------------------------------------------
# Train / validation split
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# --------------------------------------------------
# Train
# --------------------------------------------------

pipeline.fit(X_train, y_train)


# --------------------------------------------------
# Validation predictions
# --------------------------------------------------

# The model produces probabilities for the positive class:
# recoverable = class 1
probabilities = pipeline.predict_proba(X_val)[:, 1]


# --------------------------------------------------
# Validation threshold sweep
# --------------------------------------------------

# Test candidate thresholds from 0.10 through 0.90.
thresholds = np.arange(0.10, 0.91, 0.01)

threshold_results = []

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    threshold_results.append(
        {
            "threshold": round(float(threshold), 2),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    )


# --------------------------------------------------
# Save threshold analysis
# --------------------------------------------------

pd.DataFrame(threshold_results).to_csv(
    THRESHOLD_FILE,
    index=False,
)


# --------------------------------------------------
# Select threshold
# --------------------------------------------------

# Selection rule:
# 1. Choose the threshold with the highest validation F1.
# 2. If multiple thresholds have the same F1,
#    choose the lowest threshold.
best_threshold_result = max(
    threshold_results,
    key=lambda result: (
        result["f1"],
        -result["threshold"],
    ),
)

threshold = best_threshold_result["threshold"]

predictions = (probabilities >= threshold).astype(int)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_val,
    predictions,
    labels=[0, 1],
).ravel()

metrics = {
    "model": "logistic_regression",
    "target": TARGET,
    "positive_class": "recoverable",
    "negative_class": "not_recoverable",
    "threshold": threshold,
    "threshold_selection": "maximum_validation_f1",
    "threshold_tie_break": "lowest_threshold",
    "validation_rows": len(X_val),
    "true_positives": int(tp),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "recall_recoverable": float(
        recall_score(
            y_val,
            predictions,
            zero_division=0,
        )
    ),
    "precision_recoverable": float(
        precision_score(
            y_val,
            predictions,
            zero_division=0,
        )
    ),
    "f1": float(
        f1_score(
            y_val,
            predictions,
            zero_division=0,
        )
    ),
    "pr_auc": float(
        average_precision_score(
            y_val,
            probabilities,
        )
    ),
    "roc_auc": float(
        roc_auc_score(
            y_val,
            probabilities,
        )
    ),
    "brier_score": float(
        brier_score_loss(
            y_val,
            probabilities,
        )
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(
            y_val,
            predictions,
        )
    ),
}


# --------------------------------------------------
# Save model + validation metrics
# --------------------------------------------------

joblib.dump(
    pipeline,
    MODEL_FILE,
)

with METRICS_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics,
        f,
        indent=2,
    )


# --------------------------------------------------
# Threshold selection output
# --------------------------------------------------

print("\nThreshold Selection")

print(
    "Selection criterion:             "
    "Maximum validation F1"
)

print(
    "Tie-break rule:                  "
    "Lowest threshold"
)

print(
    f"Selected threshold:             "
    f"{best_threshold_result['threshold']:.2f}"
)

print(
    f"Selected recall:                "
    f"{best_threshold_result['recall']:.4f}"
)

print(
    f"Selected precision:             "
    f"{best_threshold_result['precision']:.4f}"
)

print(
    f"Selected F1:                    "
    f"{best_threshold_result['f1']:.4f}"
)


# --------------------------------------------------
# Output
# --------------------------------------------------

print("=" * 60)
print("PHASE 2 — LOGISTIC REGRESSION DETECTION")
print("=" * 60)

print(
    f"Training rows:   {len(X_train)}"
)

print(
    f"Validation rows: {len(X_val)}"
)

print(
    f"Threshold:       {threshold}"
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
    f"Metrics:          "
    f"{METRICS_FILE}"
)

print(
    f"Threshold Analysis:"
    f" {THRESHOLD_FILE}"
)

print("\nTraining completed.")
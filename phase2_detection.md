# Phase 2 — Revenue Risk Detection

## Objective

Predict whether a revenue-risk event is:

- `recoverable`
- `not_recoverable`

## Target

`risk`

Positive class:

`recoverable`

Negative class:

`not_recoverable`

## Baseline Model

Logistic Regression.

This is a baseline model, not a final model selection.

## Input Features

- `event`
- `amount`
- `failure`
- `payment_method`
- `prior_failed_attempts`
- `customer_risk_score`
- `time_since_last_purchase_days`
- `historical_ltv_inr`
- `days_overdue`

## Excluded Features

The following will not be used as detection features:

- `record_id`
- `customer_id`
- `merchant_id`
- `risk`
- `root_cause`
- `agent_decision`
- `action`
- `attempt`
- `result`
- `recovered`
- `status`
- `stop_reason`
- `p_recover_true`

## Data Split

The frozen `test.csv` will remain untouched until final evaluation.

`train.csv` will be split into:

- training set
- validation set

The validation set will be used for model/threshold decisions.

The test set will be used once for final evaluation.

## Primary Metrics

- Recall for `recoverable`
- Precision for `recoverable`

## Supporting Metrics

- PR-AUC
- F1
- ROC-AUC
- Confusion matrix
- Brier score

## Threshold

The classification threshold will be selected using the validation set.

The test set will not be used for threshold tuning.

## Phase 2 Goal

Establish a reproducible Logistic Regression detection baseline and evaluate it without data leakage.
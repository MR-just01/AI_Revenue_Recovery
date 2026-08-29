# AI Revenue Recovery — Dataset Schema

**Phase:** 1 — Freeze and Validate Dataset/Schema  
**Dataset version:** Phase 1 baseline  
**Source files:** `revenue_recovery_full.csv`, `train.csv`, `test.csv`

## 1. Dataset overview

The dataset represents synthetic end-to-end revenue-recovery records.

- Full dataset: **600 records**
- Training set: **476 records**
- Test set: **124 records**
- Total columns: **23**
- Train + test = **600**
- `record_id` must be unique across the full dataset.
- Train/test records must not overlap by `record_id`.

This schema is the contract for Phase 1. Later phases must not silently change column meaning or use downstream outcome fields as model inputs.

---

## 2. Column classification

### A. Event / context fields

| Column | Type | Meaning | ML role |
|---|---|---|---|
| `record_id` | string | Unique recovery-event identifier. | Identifier; exclude from model features. |
| `customer_id` | string | Customer identifier. | Identifier; exclude from model features unless a later design explicitly derives historical aggregates without leakage. |
| `merchant_id` | string | Merchant identifier. | Identifier/context; exclude from baseline model features. |
| `created_at` | datetime string | Time at which the revenue-risk event was created. | Available at prediction time; baseline models should derive time features only if explicitly implemented. |
| `event` | categorical | Revenue-risk event type. | Input feature. |
| `amount` | integer | Amount of revenue associated with the event, in INR. | Input feature. |
| `failure` | categorical | Failure/drop-off/overdue reason associated with the event. | Input feature. |
| `payment_method` | categorical | Payment method associated with the event. | Input feature. |
| `prior_failed_attempts` | integer | Number of prior failed payment attempts available at event time. | Input feature. |
| `customer_risk_score` | float | Synthetic customer risk score available at event time. | Input feature. |
| `time_since_last_purchase_days` | integer | Days since the customer's last purchase. | Input feature. |
| `historical_ltv_inr` | float | Historical customer lifetime value in INR. | Input feature. |
| `days_overdue` | float / null | Number of days an invoice is overdue. Applicable to `overdue_invoice` events; null for other event types. | Input feature with event-specific missingness. |

### B. ML targets

| Column | Type | Meaning | ML role |
|---|---|---|---|
| `risk` | categorical | Whether the event is labelled recoverable or not recoverable. | **Detection target.** |
| `root_cause` | categorical | Labelled reason for why the revenue is at risk. | **Diagnosis target.** |

### C. Agent / policy fields

| Column | Type | Meaning | ML role |
|---|---|---|---|
| `agent_decision` | categorical | Decision selected by the recovery policy. | Downstream policy output; exclude from Detection/Diagnosis inputs. |
| `action` | categorical | Concrete recovery action associated with the decision. | Downstream action; exclude from Detection/Diagnosis inputs. |

### D. Execution / outcome fields

| Column | Type | Meaning | ML role |
|---|---|---|---|
| `attempt` | categorical string | Attempt number and maximum allowed attempts, e.g. `1/3`, `2/3`, `0/0`. | Execution trace; exclude from Detection/Diagnosis inputs. |
| `result` | categorical | Execution result: success, failure, or not attempted. | Outcome; exclude from Detection/Diagnosis inputs. |
| `recovered` | float | Amount recovered by the simulated/test execution, in INR. | Business outcome; exclude from Detection/Diagnosis inputs. |
| `status` | categorical | Final recovery state. | Outcome; exclude from Detection/Diagnosis inputs. |
| `stop_reason` | categorical | Reason the recovery workflow stopped. | Outcome/control trace; exclude from Detection/Diagnosis inputs. |

### E. Synthetic generator field

| Column | Type | Meaning | ML role |
|---|---|---|---|
| `p_recover_true` | float | Synthetic generator-side underlying recovery probability. | **Debug/ground-truth field; never use as an ML feature.** |

---

## 3. Observed categorical values

### `event`

- `payment_failure`
- `checkout_abandonment`
- `subscription_failure`
- `overdue_invoice`

### `failure`

- `payment_selection_dropoff`
- `overdue_31_60d`
- `address_dropoff`
- `overdue_1_30d`
- `insufficient_funds`
- `issuer_down`
- `otp_dropoff`
- `mandate_limit_exceeded`
- `otp_failure`
- `3ds_failure`
- `network_error`
- `risk_declined`
- `card_expired`
- `cart_dropoff`
- `bank_timeout`
- `overdue_60d_plus`

### `root_cause`

- `customer_hesitation`
- `hard_decline`
- `soft_decline`
- `technical_issue`

### `risk`

- `recoverable`
- `not_recoverable`

### `agent_decision`

- `send_reminder`
- `write_off`
- `soft_reminder_low_priority`
- `retry_alternate_method`
- `retry_later`
- `retry_prompt`
- `retry_now`
- `send_payment_reminder`
- `escalate_to_collections`
- `offer_alternate_payment`

### `action`

- `send_whatsapp_reminder`
- `no_action`
- `send_email_reminder`
- `retry_payment_alt_method`
- `retry_payment`
- `send_retry_link`
- `send_invoice_reminder`
- `escalate_to_human`
- `send_payment_link`
- `send_sms_reminder`

### `result`

- `success`
- `not_attempted`
- `failure`

### `status`

- `recovered`
- `not_recovered`
- `failed`
- `stopped`
- `escalated`

### `stop_reason`

- `customer_paid`
- `predicted_not_recoverable_skipped`
- `max_retries_reached`
- `payment_success`
- `customer_opted_out`
- `escalated_unresolved`

### `payment_method`

- `netbanking`
- `invoice`
- `upi`
- `card`
- `emi`
- `wallet`

---

## 4. Numeric field constraints observed in the Phase 1 dataset

These are validation ranges observed in the frozen baseline, not universal business rules:

| Column | Observed range |
|---|---:|
| `amount` | 50 to 128450 INR |
| `prior_failed_attempts` | 0 to 4 |
| `customer_risk_score` | 0.011 to 0.844 |
| `time_since_last_purchase_days` | 0 to 399 |
| `historical_ltv_inr` | 53.13 to 76414.76 INR |
| `days_overdue` | 1 to 89 when applicable |
| `p_recover_true` | 0.03 to 0.97 |
| `recovered` | 0 to 128450 INR |

Validation must check that `recovered <= amount`.

---

## 5. Missing-value policy

`days_overdue` has event-specific missingness:

- For `overdue_invoice`, the field is expected to be populated.
- For non-invoice events, the field may be null.

Do **not** blindly replace these nulls with zero. The validation script should verify the event-specific rule.

Other required fields in the Phase 1 baseline should not contain unexpected null values.

---

## 6. ML feature contract

### Detection model

**Target:**

`risk`

**Baseline allowed input features:**

```text
event
amount
failure
payment_method
prior_failed_attempts
customer_risk_score
time_since_last_purchase_days
historical_ltv_inr
days_overdue
```

Identifiers are not baseline features.

The following must NOT be used as Detection features:

```text
risk
root_cause
agent_decision
action
attempt
result
recovered
status
stop_reason
p_recover_true
```

### Diagnosis model

**Target:**

`root_cause`

Use information available before the diagnosis/action decision. Do not use downstream action or outcome fields.

The exact Diagnosis feature set must be documented in the Phase 2 model code before training.

---

## 7. Temporal / outcome boundary

The core leakage rule is:

> A model may only use information that would be available at the point in the workflow where the prediction is made.

Therefore:

```text
Event/context
    ↓
Detection: predict risk
    ↓
Diagnosis: predict root_cause
    ↓
Policy: choose decision/action
    ↓
Execution: attempt action
    ↓
Outcome: result/recovered/status/stop_reason
```

Fields produced after an action must not be fed backward into Detection or Diagnosis.

---

## 8. Recovery consistency rules

The validation script should enforce at least:

1. `recovered >= 0`
2. `recovered <= amount`
3. `result == success` implies a successful recovery outcome is possible, but the exact status mapping must follow the generator's defined policy.
4. `recovered > 0` must correspond to a recovered outcome in the current baseline.
5. `attempt` must follow the `N/M` format used by the dataset.
6. `0/0` represents a non-attempted case in the current baseline.
7. `stop_reason` must belong to the allowed set.
8. `status` must belong to the allowed set.

These checks validate the dataset; they do not redefine the generator's behavior.

---

## 9. Phase 1 freeze rule

Once validation passes, the following are frozen for the Phase 1 checkpoint:

- column names
- column meanings
- target definitions
- allowed categorical values
- missing-value semantics
- train/test membership
- leakage exclusions
- recovery consistency rules

Any future schema change must be made deliberately in a later branch/version and documented.

---

## 10. Phase 1 acceptance criteria

Phase 1 is accepted only when:

- [ ] Full dataset contains 600 records.
- [ ] Train contains 476 records.
- [ ] Test contains 124 records.
- [ ] Train + test account for all full-dataset records.
- [ ] No train/test `record_id` overlap.
- [ ] `record_id` is unique in the full dataset.
- [ ] Required fields contain no unexpected nulls.
- [ ] `days_overdue` follows its event-specific missingness rule.
- [ ] Categorical values are from the documented sets.
- [ ] Numeric fields pass sanity/range checks.
- [ ] `recovered <= amount` for every record.
- [ ] Downstream outcome fields are excluded from ML feature definitions.
- [ ] `p_recover_true` is excluded from ML features.
- [ ] Validation script passes.
- [ ] This schema and the exact dataset files are committed to `phase-1-data-validation`.

**Phase 1 status:** Schema defined; final freeze occurs only after the validation script passes.

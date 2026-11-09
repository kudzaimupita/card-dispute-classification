# Annotation guidelines

These guidelines govern the human-labelled test set. Label definitions, inclusions,
exclusions and boundary rules are maintained in [`configs/labels.yaml`](../configs/labels.yaml);
this document describes the procedure.

## Task

Read one customer message and assign exactly one of five labels:

| Label | Short name |
|---|---|
| `fraud` | Unauthorised transaction |
| `goods_not_received` | Goods or services not received |
| `duplicate_charge` | Duplicate or incorrect amount |
| `subscription_cancellation` | Subscription or recurring payment |
| `other` | Other card or payment request |

## Procedure

1. Annotate blind. The annotation sheet shows the message text only: no generator label,
   attributes, model name or prediction.
2. Read the whole message, then ask: *what does the customer want the bank to do now?*
   Label that primary request.
3. If two labels seem possible, apply the boundary rules in `labels.yaml` in order.
4. If no rule resolves it, choose the label you find more likely and set `uncertain = 1`.
   Do not skip items.
5. Work in sessions of at most 50 messages to limit fatigue.

## Two-annotator protocol

- Annotator A labels all 200 test messages; annotator B labels at least 100, sampled
  uniformly by item id.
- Annotators do not discuss items before both have finished.
- Agreement is reported as Cohen's kappa and Krippendorff's alpha with bootstrap
  confidence intervals, overall and per label.
- Disagreements are adjudicated in one joint session. The adjudicated label becomes gold;
  pre-adjudication labels are kept for the agreement statistics.
- Any rule added during adjudication is written back into `labels.yaml` with the date, and
  the change is listed in the adjudication log.

## Annotator record

For each annotator, record: role or background, prior exposure to payment disputes,
date of annotation, and consent to be acknowledged. No annotator sees employer data.

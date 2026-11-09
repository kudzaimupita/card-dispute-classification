# Development log

Dated record of decisions that affect results. One entry per decision.

## 2026-11-09

- Pipeline built: label schema, attribute grid, batch generation, three-stage deduplication,
  post-hoc typo noise, group split, TF-IDF baseline with grouped cross-validation.
- First generation batch: 450 requests to claude-sonnet-5-5 at effort low; 450 succeeded,
  0 failed; 3,587 messages (13 requests returned fewer than eight). Deduplication removed
  1 exact, 1 near and 8 semantic duplicates. No semantic overlap between train and the
  validation or test splits.
- TF-IDF + logistic regression: grouped CV macro-F1 0.979 to 0.986 across C; validation
  macro-F1 0.997 [0.991, 1.000], one error in 358.
- Diagnosis: messages were realistic and explicit cue words appeared in only 22 to 38 % of
  fraud, goods-not-received and duplicate-charge messages, but every message was a typical
  case because the prompt asked for clear membership. Real disputes are hard when phrased
  indirectly or when they mix issues.
- Decision: add a difficulty attribute (typical, indirect, multi-issue) and regenerate.
  Recorded in the pre-registration. First corpus kept locally as `data/v1-typical/`.

# Pre-registration

Written 9 November 2026, before any data were generated or any model was trained.
Changes after this date are listed at the end with their reason.

## Research question

On a five-class card-dispute message task, how do a classical baseline, a fine-tuned
encoder and zero-shot large language models compare in accuracy, calibration and cost
per 1,000 messages, and where does each fail?

## Hypotheses

- **H1.** A fine-tuned encoder (DeBERTa-v3-base) achieves higher macro-F1 than TF-IDF with
  logistic regression on the human-labelled test set.
- **H2.** Zero-shot frontier LLMs reach within 5 macro-F1 points of the fine-tuned encoder,
  at a cost per 1,000 messages at least one order of magnitude higher.
- **H3.** All systems lose macro-F1 on messages with heavy typo noise relative to clean
  messages, and the classical character n-gram model loses least.

## Data

- Training and validation data: synthetic messages generated under a controlled attribute
  grid (`configs/attributes.yaml`), deduplicated, split by generation slot so that
  messages from one request never appear in two splits.
- Primary test set: 200 messages from a separately seeded generation batch, labelled blind
  by two annotators (`docs/annotation-guidelines.md`).
- Secondary test set: real banking queries from BANKING77 mapped to the five labels. This
  set is reported separately because public benchmarks may appear in LLM pre-training data.

## Primary metric and analysis

- Primary metric: macro-F1 on the human-labelled test set.
- Secondary: accuracy, per-class F1, expected calibration error (15 bins), Brier score,
  cost per 1,000 messages, median and 95th percentile latency.
- Uncertainty: 1,000-resample paired bootstrap over test items for every metric and every
  pairwise difference; McNemar's exact test for pairwise accuracy; Holm correction across
  the family of pairwise tests. Seed variance (5 seeds) is reported separately.
- Model selection uses the validation split only. Each final configuration is evaluated on
  the test sets once.

## Power

With 200 test items the 95% confidence interval on macro-F1 is expected to be about
±5 points. Differences smaller than that are reported as inconclusive.

## Stopping rule

No additional data are generated after the test set has been labelled. Hyperparameter
search is limited to the grids stated in the configuration files.

## Changes after registration

None yet.

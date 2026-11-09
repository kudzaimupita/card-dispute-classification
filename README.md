# Card-dispute message classification

How well do a classical baseline, a fine-tuned encoder and zero-shot large language models
classify customer messages about card disputes, and where does each fail?

The task has five classes: unauthorised transaction, goods not received, duplicate charge,
subscription or recurring payment, and other. Training data is a synthetic corpus generated
under a controlled attribute grid; the primary test set is labelled blind by two human
annotators. The study design, hypotheses and analysis plan are fixed in advance in
[`docs/preregistration.md`](docs/preregistration.md).

## Status

| Stage | State |
|---|---|
| Label schema, annotation guidelines, pre-registration | Done |
| Generation pipeline (planning, prompts, batch submit and collect) | Done |
| Deduplication, typo noise, group split | Done |
| TF-IDF + logistic regression baseline | Done |
| Corpus generation run | Pending |
| Human-labelled test set and agreement | Pending |
| Fine-tuned encoders (DeBERTa-v3-base, UltraBERT-150M) | Pending |
| Zero-shot LLM baselines with cost accounting | Pending |
| Calibration, significance testing, error analysis, report | Pending |

## Method in brief

1. **Labels.** Definitions, inclusions, exclusions and boundary rules for multi-issue messages
   live in [`configs/labels.yaml`](configs/labels.yaml), the single source for prompts and
   annotation.
2. **Generation.** Each request asks for eight messages of one label under one sampled
   profile: persona, channel, length, variety of English, currency, merchant type and tone
   ([`configs/attributes.yaml`](configs/attributes.yaml)). Requests run as one Message Batch
   with a closed JSON output schema. Every message records its model, batch, request and
   prompt hash.
3. **Deduplication.** Exact match after normalisation, MinHash near-duplicates on character
   5-grams, then embedding similarity. Every removal is logged with the message it duplicates.
4. **Noise.** Typos are injected after deduplication by a seeded keyboard-error model, so the
   noise is identical for every generator and can be ablated. Numbers and amounts are never
   altered.
5. **Splits.** All messages from one request stay in one split, and splits are stratified by
   label. The pipeline fails if any request crosses a split, and reports semantic overlap
   between training and evaluation splits.
6. **Evaluation.** Macro-F1 is the primary metric, with 1,000-resample bootstrap intervals and
   paired bootstrap intervals for differences between systems.

## Repository layout

```
configs/          label schema, attribute grid and per-stage settings
docs/             pre-registration and annotation guidelines
src/dispute_clf/
  generation/     slot planning, prompts, batch submission and collection
  preprocessing/  deduplication, typo injection, group split
  baselines/      TF-IDF + logistic regression
  evaluation/     metrics and bootstrap intervals
  cli.py          command-line entry point
tests/            unit tests and an end-to-end pipeline test
```

## Reproducing

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
make install
make check
```

Generating the corpus calls the Anthropic Message Batches API. Copy `.env.example` to `.env`
and set `ANTHROPIC_API_KEY`, then:

```bash
make plan      # sample 450 generation slots
make submit    # submit one batch (asks for confirmation)
make collect   # wait for the batch and write data/raw/generated.jsonl
make pipeline  # deduplicate, add noise, split, train and score the baseline
```

Every stage reads its settings from `configs/` and writes to the paths named there; see
[`data/README.md`](data/README.md) for the data layout.

## Data and licensing

Code is released under the Apache License 2.0. The corpus is entirely synthetic: all names,
merchants, amounts and references are invented, and no real customer or employer data is
used at any stage.

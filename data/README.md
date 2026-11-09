# Data

Generated and derived data live here and are not committed. Recreate them with the
pipeline (see the top-level README). The released corpus is published separately with
its datasheet.

| Path | Produced by | Contents |
|---|---|---|
| `interim/slots.jsonl` | `dispute-clf generate plan` | Generation slots: label, attributes, seed |
| `interim/batch.json` | `dispute-clf generate submit` | Batch id and submission metadata |
| `raw/generated.jsonl` | `dispute-clf generate collect` | Generated messages with provenance |
| `processed/clean.jsonl` | `dispute-clf dedup` | Deduplicated corpus with typo noise applied |
| `processed/{train,val,test}.jsonl` | `dispute-clf split` | Label-stratified group splits |

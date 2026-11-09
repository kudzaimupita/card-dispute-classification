"""Command-line entry point: ``dispute-clf <command>``.

Each command reads its settings from a YAML file under ``configs/`` and writes its outputs
to the paths named there, so every stage can be re-run independently.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import anthropic
import typer
from dotenv import load_dotenv

from dispute_clf.attributes import load_attribute_grid
from dispute_clf.baselines import tfidf
from dispute_clf.evaluation.metrics import Scores, score
from dispute_clf.external import ExternalExample, build_mapped_set, load_intent_mapping
from dispute_clf.generation.batch import BatchRecord, build_requests, collect, submit
from dispute_clf.generation.config import GenerationConfig, load_generation_config
from dispute_clf.generation.planner import GenerationSlot, plan_slots
from dispute_clf.generation.prompts import build_system_prompt, build_user_prompt
from dispute_clf.labels import load_label_schema
from dispute_clf.preprocessing.dedup import cross_split_overlap, deduplicate
from dispute_clf.preprocessing.split import Split, leaked_groups, split_by_group
from dispute_clf.preprocessing.typos import apply_typos
from dispute_clf.records import LabelledText, Message, read_jsonl, write_jsonl
from dispute_clf.settings import load_preprocessing_config, load_tfidf_config

app = typer.Typer(no_args_is_help=True, help="Card-dispute message classification pipeline.")
generate_app = typer.Typer(no_args_is_help=True, help="Generate the synthetic corpus.")
app.add_typer(generate_app, name="generate")

GenerationConfigOption = Annotated[Path, typer.Option("--config", help="Generation settings.")]
DEFAULT_GENERATION = Path("configs/generation.yaml")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _client() -> anthropic.Anthropic:
    load_dotenv()
    return anthropic.Anthropic()


def _prompts(config: GenerationConfig, slots: list[GenerationSlot]) -> tuple[str, dict[str, str]]:
    schema = load_label_schema(config.paths.labels)
    grid = load_attribute_grid(config.paths.attributes)
    system = build_system_prompt(schema)
    return system, {slot.slot_id: build_user_prompt(slot, schema, grid) for slot in slots}


@generate_app.command("plan")
def generate_plan(config_path: GenerationConfigOption = DEFAULT_GENERATION) -> None:
    """Sample attribute combinations and write the generation slots."""
    config = load_generation_config(config_path)
    slots = plan_slots(
        load_attribute_grid(config.paths.attributes),
        slots_per_label=config.slots_per_label,
        messages_per_slot=config.messages_per_slot,
        seed=config.seed,
    )
    count = write_jsonl(config.paths.slots, slots)
    typer.echo(
        f"{count} slots -> {config.paths.slots} ({count * config.messages_per_slot} messages)"
    )


@generate_app.command("submit")
def generate_submit(
    config_path: GenerationConfigOption = DEFAULT_GENERATION,
    yes: Annotated[bool, typer.Option("--yes", help="Submit without confirmation.")] = False,
) -> None:
    """Submit one Message Batch with a request per slot."""
    config = load_generation_config(config_path)
    if config.paths.batch.exists():
        raise typer.BadParameter(f"{config.paths.batch} exists; a batch was already submitted")
    slots = read_jsonl(config.paths.slots, GenerationSlot)
    requests = build_requests(
        slots,
        config,
        load_label_schema(config.paths.labels),
        load_attribute_grid(config.paths.attributes),
    )
    if not yes:
        typer.confirm(f"Submit {len(requests)} requests to {config.model}?", abort=True)
    record = submit(_client(), requests, submitted_at=datetime.now(UTC))
    _write_json(config.paths.batch, record.model_dump(mode="json"))
    typer.echo(f"Submitted {record.batch_id} ({record.n_requests} requests)")


@generate_app.command("collect")
def generate_collect(
    config_path: GenerationConfigOption = DEFAULT_GENERATION,
    wait: Annotated[bool, typer.Option(help="Poll until the batch has ended.")] = True,
    poll_seconds: Annotated[int, typer.Option(min=10)] = 60,
) -> None:
    """Download batch results and write labelled messages and failures."""
    config = load_generation_config(config_path)
    record = BatchRecord.model_validate_json(config.paths.batch.read_text(encoding="utf-8"))
    client = _client()

    batch = client.messages.batches.retrieve(record.batch_id)
    while batch.processing_status != "ended":
        if not wait:
            typer.echo(f"{batch.processing_status}: {batch.request_counts}")
            raise typer.Exit(code=1)
        time.sleep(poll_seconds)
        batch = client.messages.batches.retrieve(record.batch_id)

    slots = read_jsonl(config.paths.slots, GenerationSlot)
    system, users = _prompts(config, slots)
    outcome = collect(
        client.messages.batches.results(record.batch_id),
        {slot.slot_id: slot for slot in slots},
        batch=record,
        system_prompt=system,
        user_prompts=users,
    )
    write_jsonl(config.paths.generated, outcome.messages)
    write_jsonl(config.paths.failures, outcome.failures)
    reasons = Counter(failure.reason for failure in outcome.failures)
    typer.echo(
        f"{len(outcome.messages)} messages, {len(outcome.failures)} failed slots {dict(reasons)}"
    )


@app.command("clean")
def clean(
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/preprocessing.yaml"),
) -> None:
    """Deduplicate generated messages, then apply each message's typo level."""
    config = load_preprocessing_config(config_path)
    messages = read_jsonl(config.paths.generated, Message)
    result = deduplicate(
        messages,
        near_threshold=config.near_threshold,
        semantic_threshold=config.semantic_threshold,
    )
    noisy = apply_typos(result.kept, seed=config.noise_seed)
    write_jsonl(config.paths.clean, noisy)

    label_of = {message.id: message.label.value for message in messages}
    removed_by_label = Counter(label_of[removal.id] for removal in result.removed)
    report = {
        "input": len(messages),
        "kept": len(result.kept),
        "removed": {stage: result.count(stage) for stage in ("exact", "near", "semantic")},
        "removed_by_label": dict(sorted(removed_by_label.items())),
        "kept_by_label": dict(sorted(Counter(m.label.value for m in result.kept).items())),
        "removals": [removal.model_dump() for removal in result.removed],
    }
    _write_json(config.paths.dedup_report, report)
    typer.echo(f"kept {len(result.kept)} of {len(messages)}; removed {report['removed']}")


@app.command("split")
def split(
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/preprocessing.yaml"),
) -> None:
    """Split the clean corpus by generation group and check for leakage."""
    config = load_preprocessing_config(config_path)
    messages = read_jsonl(config.paths.clean, Message)
    splits = split_by_group(
        messages,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.split_seed,
    )
    if leaked := leaked_groups(splits):
        raise typer.BadParameter(f"groups leaked across splits: {sorted(leaked)[:5]}")
    for name, items in splits.items():
        write_jsonl(config.paths.splits_dir / f"{name}.jsonl", items)

    train = splits[Split.TRAIN]
    report = {
        "sizes": {name.value: len(items) for name, items in splits.items()},
        "labels": {
            name.value: dict(sorted(Counter(m.label.value for m in items).items()))
            for name, items in splits.items()
        },
        "semantic_overlap_with_train": {
            name.value: cross_split_overlap(
                train, splits[name], threshold=config.semantic_threshold
            )
            for name in (Split.VAL, Split.TEST)
        },
    }
    _write_json(config.paths.split_report, report)
    typer.echo(f"sizes {report['sizes']}")


@app.command("build-external")
def build_external(
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/banking77.yaml"),
) -> None:
    """Map a public intent dataset onto the five labels for real-text evaluation."""
    mapping = load_intent_mapping(config_path)
    examples = build_mapped_set(mapping)
    write_jsonl(mapping.output, examples)
    counts = Counter(example.label.value for example in examples)
    typer.echo(f"{len(examples)} examples -> {mapping.output} {dict(sorted(counts.items()))}")


def _save_evaluation(
    output: Path,
    name: str,
    *,
    examples: Sequence[LabelledText],
    predictions: list[str],
    scores: Scores,
    extra: dict[str, object],
) -> None:
    _write_json(
        output / f"metrics_{name}.json", {**extra, "split": name, "scores": scores.model_dump()}
    )
    with (output / f"predictions_{name}.jsonl").open("w", encoding="utf-8") as handle:
        for example, predicted in zip(examples, predictions, strict=True):
            row = {"id": example.id, "label": example.label.value, "pred": predicted}
            handle.write(json.dumps(row) + "\n")
    f1 = scores.macro_f1
    typer.echo(f"{name}: macro-F1 {f1.estimate:.3f} [{f1.low:.3f}, {f1.high:.3f}] (n={scores.n})")


@app.command("train-tfidf")
def train_tfidf(
    config_path: Annotated[Path, typer.Option("--config")] = Path("configs/tfidf.yaml"),
    evaluate_on: Annotated[Split, typer.Option(help="Split to report.")] = Split.VAL,
    external: Annotated[
        Path | None, typer.Option(help="Also score a mapped real-text set (JSONL).")
    ] = None,
) -> None:
    """Select C by grouped cross-validation on train and score the chosen split."""
    config = load_tfidf_config(config_path)
    train = read_jsonl(config.paths.splits_dir / "train.jsonl", Message)
    fitted = tfidf.fit(
        train,
        c_grid=config.c_grid,
        n_splits=config.cv_folds,
        min_df=config.min_df,
        seed=config.seed,
    )
    meta: dict[str, object] = {
        "system": "tfidf_logreg",
        "best_c": fitted.best_c,
        "cv_macro_f1": fitted.cv_macro_f1,
    }
    typer.echo(f"C={fitted.best_c}")

    evaluations: list[tuple[str, Sequence[LabelledText]]] = [
        (
            evaluate_on.value,
            read_jsonl(config.paths.splits_dir / f"{evaluate_on.value}.jsonl", Message),
        )
    ]
    if external is not None:
        evaluations.append((external.stem, read_jsonl(external, ExternalExample)))

    for name, examples in evaluations:
        gold = [example.label.value for example in examples]
        predictions = tfidf.predict(fitted.pipeline, examples)
        present = sorted(set(gold))
        scores = score(
            gold,
            predictions,
            labels=present,
            n_resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        _save_evaluation(
            config.paths.output_dir,
            name,
            examples=examples,
            predictions=predictions,
            scores=scores,
            extra=meta,
        )

.DEFAULT_GOAL := help
RUN := uv run

.PHONY: help install lint format typecheck test check plan submit collect clean-data split tfidf pipeline

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install dependencies from the lockfile
	uv sync --frozen

lint: ## Lint with ruff
	$(RUN) ruff check src tests

format: ## Format with ruff
	$(RUN) ruff format src tests

typecheck: ## Type-check with mypy
	$(RUN) mypy

test: ## Run the test suite
	$(RUN) pytest

check: lint typecheck test ## Lint, type-check and test

plan: ## Sample generation slots
	$(RUN) dispute-clf generate plan

submit: ## Submit the generation batch (asks for confirmation)
	$(RUN) dispute-clf generate submit

collect: ## Wait for the batch and write generated messages
	$(RUN) dispute-clf generate collect

clean-data: ## Deduplicate and apply typo noise
	$(RUN) dispute-clf clean

split: ## Group-stratified train/val/test split
	$(RUN) dispute-clf split

tfidf: ## Train and score the TF-IDF baseline on the validation split
	$(RUN) dispute-clf train-tfidf

pipeline: clean-data split tfidf ## Run every offline stage after collection

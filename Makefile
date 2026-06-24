# FinSentiment pipeline orchestration.
# Run `make help` to list targets. Stages run in order: dataset -> sentiment -> train -> backtest.

.PHONY: help setup dataset sentiment train backtest demo test clean

MODEL ?= gpt4o_mini
CLF   ?= xgboost

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package + all extras
	pip install -e ".[sentiment,data,analysis]"

dataset:  ## Build the aligned master table from raw news + prices
	python scripts/build_dataset.py

sentiment:  ## Extract sentiment for one MODEL (e.g. make sentiment MODEL=finbert)
	python scripts/run_sentiment.py --model $(MODEL)

train:  ## Train + evaluate the full model grid -> results/metrics.csv
	python scripts/train.py

backtest:  ## Sharpe-ratio backtest for MODEL + CLF
	python scripts/backtest.py --model $(MODEL) --classifier $(CLF)

demo:  ## End-to-end smoke run on the committed sample data (no keys/data needed)
	python scripts/train.py --sample --models gpt4o_mini --classifiers random_forest --verbose
	python scripts/backtest.py --sample --model gpt4o_mini --classifier random_forest

test:  ## Run the test suite
	pytest -q

clean:  ## Remove generated artifacts and caches
	rm -rf results outputs .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

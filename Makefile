.PHONY: test verify-baseline prepare-dataset validate-dataset

test:
	uv run pytest

verify-baseline:
	uv run python experiences/08/verify_runtime.py

prepare-dataset:
	uv run veridis-tille dataset prepare

validate-dataset:
	uv run veridis-tille dataset validate

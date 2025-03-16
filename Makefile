ALL=data src run.py

.PHONY: lint pretty

lint:
	ruff format --check $(ALL)
	ruff check $(ALL)
	mypy

pretty:
	ruff format $(ALL)
	ruff check --fix $(ALL)
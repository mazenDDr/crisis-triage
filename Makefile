.PHONY: check test env-check pull site

check:
	python -m ruff check .
	python -m ruff format --check .
	PYTHONPATH=src python -m pytest

test:
	PYTHONPATH=src python -m pytest

pull:
	rsync -az --exclude-from=.pullignore gpu-box:$${GPU_REMOTE_DIR:-crisis-triage}/outputs/ ./outputs/

env-check:
	PYTHONPATH=src python scripts/env_check.py

site:
	PYTHONPATH=src python scripts/build_site.py

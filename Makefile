quality:
	ruff check src/rein scripts/
	ruff format --check src/rein scripts/

style:
	ruff check src/rein scripts/ --fix
	ruff format src/rein scripts/

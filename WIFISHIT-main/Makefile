PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install test run-server

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

run-server:
	$(PYTHON) -m visualization.ui_server --host localhost --port 8000 --fps 8

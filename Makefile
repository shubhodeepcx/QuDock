# shortcuts for the usual dev loop; `make help` lists them

# qiskit-aer and scikit-learn each bundle OpenMP, see the CI workflow
TEST_ENV = GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096000 RUST_MIN_STACK=33554432 \
           OMP_STACKSIZE=64M

.PHONY: help install lint format test check app reproduce clean

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## install the package with dev tools in editable mode
	pip install -e .[dev]

lint:  ## run ruff
	ruff check .

format:  ## fix what ruff can fix by itself
	ruff check . --fix

test:  ## run the test suite
	$(TEST_ENV) pytest -q

check: lint test  ## lint and test, same as CI

app:  ## start the streamlit app
	streamlit run app/app.py

reproduce:  ## rebuild every table and figure from saved data
	qudock reproduce

clean:  ## remove caches and build leftovers
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

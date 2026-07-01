.PHONY: venv venv-gpu install-dev data detect ablate test coverage notebook

MODEL ?= meta-llama/Meta-Llama-3-8B-Instruct
DATASET ?= nolima
HEADS ?= retrieval_heads/$(notdir $(MODEL))_logit_contrib_$(DATASET).json
ARGS ?=
VENV ?= .venv
VENV_PYTHON = $(VENV)/bin/python
HOST_PYTHON ?= python3.11
GPU_PYTHON ?= python3.12
PYTHON ?= $(VENV_PYTHON)

# Local macOS venv for running unit tests (no GPU). vLLM ships no macOS wheel,
# so install the package without deps and inherit the scientific stack
# (numpy/pandas/...) from the chosen system Python via --system-site-packages.
# only-system stops uv from provisioning its own (empty) managed interpreter,
# so $(PYTHON) must be a real install that already has the stack.
# Run tests with: make test ARGS="tests/test_standalone_surface.py"
venv:
	uv venv --python $(HOST_PYTHON) --python-preference only-system --system-site-packages $(VENV)
	uv pip install -e ".[dev]" --python $(VENV_PYTHON) --no-deps
	uv pip install pytest pytest-mock pytest-asyncio pyyaml rich --python $(VENV_PYTHON)

# GPU host venv (CUDA 12.8) with the full eval stack, including vLLM.
venv-gpu:
	uv venv --python $(GPU_PYTHON) --python-preference only-system --system-site-packages $(VENV)
	uv pip install -e ".[dev,eval]" \
		--extra-index-url https://download.pytorch.org/whl/cu128 \
		--index-strategy unsafe-best-match \
		--python $(VENV_PYTHON)

install-dev:
	uv pip install -e ".[dev,eval]" --python $(PYTHON)

data:
	$(PYTHON) locos/download_haystack_data.py --dataset $(DATASET) $(ARGS)

detect:
	$(PYTHON) -m locos.detectors.logit_contrib --model $(MODEL) --dataset $(DATASET) $(ARGS)

ablate:
	$(PYTHON) -m locos_eval.evals.tasks.babilong_task --model $(MODEL) --heads $(HEADS) --decoding ablation --ablation-mode mean $(ARGS)

test:
	$(PYTHON) -m pytest $(ARGS)

coverage:
	$(PYTHON) -m pytest --cov=locos --cov=locos_eval --cov=scripts --cov-report=term-missing --cov-fail-under=90 $(ARGS)

notebook:
	$(PYTHON) -m json.tool notebooks/locos_demo.ipynb >/dev/null

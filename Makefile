.PHONY: install-dev data detect ablate test coverage notebook

MODEL ?= meta-llama/Meta-Llama-3-8B-Instruct
DATASET ?= nolima
HEADS ?= retrieval_heads/$(notdir $(MODEL))_logit_contrib_$(DATASET).json
ARGS ?=
PYTHON ?= python3.11

install-dev:
	uv pip install -e ".[dev,eval]" --python $(PYTHON)

data:
	$(PYTHON) locos/download_haystack_data.py --dataset $(DATASET) $(ARGS)

detect:
	$(PYTHON) -m locos.detectors.logit_contrib --model $(MODEL) --dataset $(DATASET) $(ARGS)

ablate:
	$(PYTHON) -m locos_eval.evals.tasks.nq_swap_task --model $(MODEL) --heads $(HEADS) --decoding ablation --ablation-mode mean $(ARGS)

test:
	$(PYTHON) -m pytest $(ARGS)

coverage:
	$(PYTHON) -m pytest --cov=locos --cov=locos_eval --cov=scripts --cov-report=term-missing --cov-fail-under=90 $(ARGS)

notebook:
	$(PYTHON) -m json.tool notebooks/locos_demo.ipynb >/dev/null

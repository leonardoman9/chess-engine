.PHONY: install build shell train-local train-hydra-small train-hydra-medium train-hydra-large train-hydra-gpu train-hydra-custom train-hydra-cnn-gpu0 analyze-training analyze-games evaluate-elo-quick evaluate-elo-full tensorboard notebook clean help

PYTHON ?= python3
DOCKER_COMPOSE ?= docker compose
SERVICE ?= chess-rl

# ========= Environment setup =========
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

build:
	$(DOCKER_COMPOSE) build

shell:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) /bin/bash

# ========= Training =========
train-local:
	$(PYTHON) train_hydra.py $(if $(EXP),experiment=$(EXP),) $(PARAMS)

train-hydra-small:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python train_hydra.py experiment=baseline_small

train-hydra-medium:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python train_hydra.py experiment=baseline_medium

train-hydra-large:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python train_hydra.py experiment=baseline_large

train-hydra-custom:
	@if [ -z "$(PARAMS)" ]; then echo "Usage: make train-hydra-custom PARAMS='experiment=baseline_small device=cpu'"; exit 1; fi
	$(DOCKER_COMPOSE) run --rm $(SERVICE) python train_hydra.py $(PARAMS)

train-hydra-gpu:
	@if [ -z "$(GPU)" ]; then echo "Usage: make train-hydra-gpu GPU=0 [EXP=baseline_medium] [PARAMS='experiment.total_games=2000']"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) $(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.server.yml run --rm $(SERVICE) python train_hydra.py $(if $(EXP),experiment=$(EXP),) device=cuda:0 $(PARAMS)

train-hydra-cnn-gpu0:
	CUDA_VISIBLE_DEVICES=0 $(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.server.yml run --rm $(SERVICE) python train_hydra.py experiment=cnn_gpu_long device=auto

# ========= Analysis & evaluation =========
analyze-training:
	@if [ -z "$(RESULTS)" ]; then echo "Usage: make analyze-training RESULTS=results/<run_dir>"; exit 1; fi
	$(PYTHON) analyze_training.py $(RESULTS) --plots

analyze-games:
	@if [ -z "$(RESULTS)" ]; then echo "Usage: make analyze-games RESULTS=results/<run_dir>"; exit 1; fi
	$(PYTHON) analyze_games.py $(RESULTS)

evaluate-elo-quick:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make evaluate-elo-quick CHECKPOINT=results/<run_dir>/final_model.pt"; exit 1; fi
	$(PYTHON) evaluate_elo.py $(CHECKPOINT) --quick

evaluate-elo-full:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make evaluate-elo-full CHECKPOINT=results/<run_dir>/final_model.pt"; exit 1; fi
	$(PYTHON) evaluate_elo.py $(CHECKPOINT)

# ========= Services =========
tensorboard:
	$(DOCKER_COMPOSE) run --rm --service-ports $(SERVICE) tensorboard --logdir=$(if $(LOGDIR),$(LOGDIR),logs) --host=0.0.0.0 --port=6006

notebook:
	$(DOCKER_COMPOSE) run --rm --service-ports $(SERVICE) jupyter lab --ip=0.0.0.0 --no-browser --notebook-dir=/app

# ========= Utilities =========
clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.py[co]" -delete

help:
	@echo "Chess-RL Make targets"
	@echo ""
	@echo "Setup:"
	@echo "  make install            Install Python dependencies locally"
	@echo "  make build              Build the Docker image"
	@echo ""
	@echo "Training:"
	@echo "  make train-local [EXP=baseline_small] [PARAMS='...']"
	@echo "  make train-hydra-small  Run baseline_small in Docker"
	@echo "  make train-hydra-medium Run baseline_medium in Docker"
	@echo "  make train-hydra-large  Run baseline_large in Docker"
	@echo "  make train-hydra-custom PARAMS='experiment=baseline_large device=cpu'"
	@echo "  make train-hydra-gpu GPU=0 [EXP=...] [PARAMS='...']"
	@echo "  make train-hydra-cnn-gpu0 Run cnn_gpu_long on GPU 0 in Docker"
	@echo ""
	@echo "Analysis & Evaluation:"
	@echo "  make analyze-training RESULTS=results/<run_dir>"
	@echo "  make analyze-games RESULTS=results/<run_dir>"
	@echo "  make evaluate-elo-quick CHECKPOINT=results/<run_dir>/final_model.pt"
	@echo "  make evaluate-elo-full CHECKPOINT=results/<run_dir>/final_model.pt"
	@echo ""
	@echo "Services:"
	@echo "  make tensorboard        Launch TensorBoard inside Docker"
	@echo "  make notebook           Launch JupyterLab inside Docker"
	@echo ""
	@echo "Utility:"
	@echo "  make shell              Open an interactive shell in Docker"
	@echo "  make clean              Remove Python caches"

.PHONY: install build test-phase1 train-dqn interactive tensorboard clean

# ============ LOCAL DEVELOPMENT ============
# Install dependencies
install:
	poetry install

# ============ DOCKER COMMANDS ============
# Build Docker image
build:
	docker compose build

# Test Phase 1 DQN implementation
test-phase1:
	docker compose run --rm chess-rl

# Start interactive container for development
interactive:
	docker compose run --rm chess-rl /bin/bash

# Start TensorBoard for monitoring
tensorboard:
	docker compose run --rm chess-rl tensorboard --logdir=/app/logs --host=0.0.0.0 --port=6006

# Test training setup
test-training:
	docker compose run --rm chess-rl python test_training.py

# Train DQN model (quick test)
train-quick:
	docker compose run --rm chess-rl python train_dqn.py baseline_small

# Train DQN model (development)
train-dev:
	docker compose run --rm chess-rl python train_dqn.py baseline_medium

# Train DQN model (production)
train-prod:
	docker compose run --rm chess-rl python train_dqn.py baseline_large

# List available experiments
list-experiments:
	docker compose run --rm chess-rl python train_dqn.py --list-experiments

# Legacy training command
train-dqn:
	docker compose run --rm chess-rl python train_dqn.py baseline_medium

# ============ SERVER DEPLOYMENT ============
# Commands optimized for A40 server deployment

# Build and test on server
server-setup:
	docker compose -f docker-compose.yml -f docker-compose.server.yml build
	docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl

# Run Phase 1 tests with GPU monitoring
server-test:
	docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl bash -c "nvidia-smi && python test_dqn_phase1.py"

# Start interactive session on server
server-interactive:
	docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl bash -c "nvidia-smi && /bin/bash"

# Monitor GPU usage during training
gpu-monitor:
	docker exec birds-lmannini-e3da-3 watch -n 1 nvidia-smi

# ============ GPU SELECTION ============
# Use specific GPU (0-9)

# Test on specific GPU
test-gpu:
	@if [ -z "$(GPU)" ]; then echo "Usage: make test-gpu GPU=1"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl python test_dqn_phase1.py

# Interactive session on specific GPU
interactive-gpu:
	@if [ -z "$(GPU)" ]; then echo "Usage: make interactive-gpu GPU=1"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl bash -c "nvidia-smi && /bin/bash"

# Training on specific GPU (quick)
train-gpu-quick:
	@if [ -z "$(GPU)" ]; then echo "Usage: make train-gpu-quick GPU=1"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl python train_dqn.py baseline_small

# Training on specific GPU (server experiment)
train-gpu:
	@if [ -z "$(GPU)" ]; then echo "Usage: make train-gpu GPU=1"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl python train_dqn.py server_experiment

# Training 1000 games on GPU
train-gpu-1000:
	@if [ -z "$(GPU)" ]; then echo "Usage: make train-gpu-1000 GPU=1"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl python train_dqn.py custom_1000

# Training 5000 games on GPU
train-gpu-5000:
	@if [ -z "$(GPU)" ]; then echo "Usage: make train-gpu-5000 GPU=1"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl python train_dqn.py custom_5000

# ============ HYDRA TRAINING ============
# Hydra-powered training with flexible configuration

# Train with Hydra (default config)
train-hydra:
	docker compose run --rm chess-rl python train_hydra.py

# Train with specific experiment
train-hydra-exp:
	@if [ -z "$(EXP)" ]; then echo "Usage: make train-hydra-exp EXP=baseline_small"; exit 1; fi
	docker compose run --rm chess-rl python train_hydra.py experiment=$(EXP)

# Train with custom parameters
train-hydra-custom:
	@if [ -z "$(PARAMS)" ]; then echo "Usage: make train-hydra-custom PARAMS='experiment.total_games=1000 model=large'"; exit 1; fi
	docker compose run --rm chess-rl python train_hydra.py $(PARAMS)

# Train with Hydra on GPU
train-hydra-gpu:
	@if [ -z "$(GPU)" ]; then echo "Usage: make train-hydra-gpu GPU=0 [EXP=baseline_small] [PARAMS='...']"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl python train_hydra.py $(if $(EXP),experiment=$(EXP),) device=cuda:0 $(PARAMS)

# Hydra multirun (parameter sweep)
train-hydra-sweep:
	docker compose run --rm chess-rl python train_hydra.py --multirun experiment=baseline_small,baseline_medium model=small,medium

# List available Hydra configurations
list-hydra-configs:
	docker compose run --rm chess-rl python train_hydra.py --help

# Check GPU status
gpu-status:
	nvidia-smi

# ============ CONTAINER MANAGEMENT ============
# Commands for managing the named container

# Check if container is running
container-status:
	@docker ps -f name=birds-lmannini-e3da-3

# Connect to running container
container-connect:
	docker exec -it birds-lmannini-e3da-3 /bin/bash

# View container logs
container-logs:
	docker logs birds-lmannini-e3da-3

# Stop the container
container-stop:
	docker stop birds-lmannini-e3da-3

# Remove the container
container-remove:
	docker rm birds-lmannini-e3da-3

# ============ CONTAINER SERVICES ============
# Run different services in the single container

# Start API server
start-api:
	docker compose run --rm -p 8000:8000 chess-rl uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start TensorBoard server  
start-tensorboard:
	docker compose run --rm -p 6006:6006 chess-rl tensorboard --logdir=/app/logs --host=0.0.0.0 --port=6006

# Start Jupyter notebook
start-jupyter:
	docker compose run --rm -p 8888:8888 chess-rl jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# ============ LEGACY COMMANDS ============
# Keep existing commands for backward compatibility

# Run the game against the model
play:
	poetry run python test_model.py --model content/models/checkpoint_10.pt

# Evaluate against Stockfish
evaluate:
	poetry run python evaluate_elo.py --model content/models/checkpoint_10.pt --stockfish /opt/homebrew/bin/stockfish --time 1.0 --games 10 --stockfish-elo 3000 --save

# Run with GUI
play-gui:
	poetry run python test_model.py --model content/models/checkpoint_10.pt --gui

evaluate-gui:
	poetry run python evaluate_elo.py --model content/models/checkpoint_10.pt --stockfish /opt/homebrew/bin/stockfish --time 1.0 --games 10 --stockfish-elo 3000 --save --gui

# API backend (legacy - now use start-api)
backend:
	make start-api

# Web interface (requires separate frontend setup)
web:
	@echo "Use 'make start-api' for backend and setup frontend separately"

# ============ UTILITIES ============
# Create necessary directories
setup-dirs:
	mkdir -p checkpoints logs results data/training

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true

# Clean Docker
clean-docker:
	docker compose down --volumes --remove-orphans
	docker system prune -f

# ============ MAC DEVELOPMENT ============
# Commands for Mac (without GPU)

# Test on Mac (CPU only)
test-mac:
	docker compose run --rm chess-rl python test_dqn_phase1.py

# Interactive on Mac
interactive-mac:
	docker compose run --rm chess-rl /bin/bash

# Clean orphaned containers
clean-orphans:
	docker compose down --remove-orphans

# Analyze game quality
analyze-games:
	docker compose run --rm chess-rl python analyze_games.py --games 5

# Analyze with checkpoint
analyze-checkpoint:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make analyze-checkpoint CHECKPOINT=path/to/checkpoint.pt"; exit 1; fi
	docker compose run --rm chess-rl python analyze_games.py --games 5 --load-checkpoint $(CHECKPOINT)

# Analyze training results
analyze-training:
	@if [ -z "$(RESULTS)" ]; then echo "Usage: make analyze-training RESULTS=results/experiment_dir"; exit 1; fi
	docker compose run --rm chess-rl python analyze_training.py $(RESULTS)

# Generate training plots
training-plots:
	@if [ -z "$(RESULTS)" ]; then echo "Usage: make training-plots RESULTS=results/experiment_dir"; exit 1; fi
	docker compose run --rm chess-rl python analyze_training.py $(RESULTS) --plots

# Generate sample games from checkpoint
sample-games:
	@if [ -z "$(RESULTS)" ]; then echo "Usage: make sample-games RESULTS=results/experiment_dir [CHECKPOINT=checkpoint.pt]"; exit 1; fi
	docker compose run --rm chess-rl python analyze_training.py $(RESULTS) --generate-games $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

# ============ ELO EVALUATION COMMANDS ============
# Evaluate ELO rating against multiple Stockfish levels

# Quick ELO evaluation (fewer games, faster)
evaluate-elo-quick:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make evaluate-elo-quick CHECKPOINT=checkpoints/latest.pt"; exit 1; fi
	docker compose run --rm chess-rl python evaluate_elo.py $(CHECKPOINT) --quick

# Full ELO evaluation (comprehensive testing)
evaluate-elo-full:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make evaluate-elo-full CHECKPOINT=checkpoints/latest.pt"; exit 1; fi
	docker compose run --rm chess-rl python evaluate_elo.py $(CHECKPOINT)

# Evaluate random baseline (no checkpoint needed)
evaluate-elo-baseline:
	docker compose run --rm chess-rl python evaluate_elo.py --quick

# ELO evaluation on GPU server
evaluate-elo-gpu:
	@if [ -z "$(GPU)" ] || [ -z "$(CHECKPOINT)" ]; then echo "Usage: make evaluate-elo-gpu GPU=0 CHECKPOINT=checkpoints/latest.pt [QUICK=1]"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm chess-rl python evaluate_elo.py $(CHECKPOINT) $(if $(QUICK),--quick,)

# Help
help:
	@echo "Chess-RL Training Commands:"
	@echo ""
	@echo "🐳 MAIN COMMANDS:"
	@echo "  make build          - Build Docker image"
	@echo "  make test-phase1    - Test Phase 1 DQN implementation"
	@echo "  make test-training  - Test training setup"
	@echo "  make interactive    - Start interactive container"
	@echo ""
	@echo "🎯 TRAINING COMMANDS:"
	@echo "  make list-experiments - List available experiments"
	@echo "  make train-quick    - Quick training test (100 games)"
	@echo "  make train-dev      - Development training (500 games)"
	@echo "  make train-prod     - Production training (2000 games)"
	@echo ""
	@echo "🔧 HYDRA TRAINING:"
	@echo "  make train-hydra    - Train with Hydra (default config)"
	@echo "  make train-hydra-exp EXP=baseline_small - Train specific experiment"
	@echo "  make train-hydra-custom PARAMS='...' - Train with custom parameters"
	@echo "  make train-hydra-gpu GPU=0 [EXP=...] - Train with Hydra on GPU"
	@echo "  make train-hydra-sweep - Parameter sweep"
	@echo "  make list-hydra-configs - List Hydra configurations"
	@echo ""
	@echo "🖥️  SERVER COMMANDS:"
	@echo "  make server-setup   - Setup and test on server"
	@echo "  make server-test    - Test with GPU monitoring"
	@echo "  make server-interactive - Interactive session on server"
	@echo ""
	@echo "🎯 GPU SELECTION:"
	@echo "  make test-gpu GPU=1        - Test on specific GPU"
	@echo "  make interactive-gpu GPU=2 - Interactive on specific GPU"
	@echo "  make train-gpu GPU=0       - Train on specific GPU"
	@echo "  make gpu-status            - Show GPU status"
	@echo ""
	@echo "🚀 SERVICES:"
	@echo "  make start-api        - Start API server (port 8000)"
	@echo "  make start-tensorboard - Start TensorBoard (port 6006)"
	@echo "  make start-jupyter    - Start Jupyter notebook (port 8888)"
	@echo ""
	@echo "🐋 CONTAINER MANAGEMENT:"
	@echo "  make container-status      - Check container status"
	@echo "  make container-connect     - Connect to running container"
	@echo "  make container-logs        - View container logs"
	@echo "  make container-stop        - Stop container"
	@echo "  make container-remove      - Remove container"
	@echo ""
	@echo "🍎 MAC DEVELOPMENT:"
	@echo "  make test-mac       - Test on Mac (CPU only)"
	@echo "  make interactive-mac - Interactive on Mac"
	@echo "  make analyze-games  - Analyze game quality"
	@echo "  make clean-orphans  - Clean orphaned containers"
	@echo ""
	@echo "🏆 ELO EVALUATION:"
	@echo "  make evaluate-elo-quick CHECKPOINT=file   - Quick ELO evaluation"
	@echo "  make evaluate-elo-full CHECKPOINT=file    - Full ELO evaluation"
	@echo "  make evaluate-elo-baseline                 - Evaluate random baseline"
	@echo "  make evaluate-elo-gpu GPU=0 CHECKPOINT=file - ELO evaluation on GPU"
	@echo ""
	@echo "🧹 UTILITIES:"
	@echo "  make clean          - Clean Python cache"
	@echo "  make clean-docker   - Clean Docker containers"
	@echo "  make help           - Show this help" 
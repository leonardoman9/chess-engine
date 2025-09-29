.PHONY: install build test-phase1 train-dqn interactive tensorboard clean

# ============ LOCAL DEVELOPMENT ============
# Install dependencies
install:
	poetry install

# ============ DOCKER COMMANDS ============
# Build Docker image
build:
	docker-compose build

# Test Phase 1 DQN implementation
test-phase1:
	docker-compose run --rm chess-dqn-training

# Start interactive container for development
interactive:
	docker-compose run --rm chess-dqn-interactive

# Start TensorBoard for monitoring
tensorboard:
	docker-compose up tensorboard

# Start full training pipeline (when ready)
train-dqn:
	docker-compose run --rm chess-dqn-training python src/training/train_dqn.py

# ============ SERVER DEPLOYMENT ============
# Commands optimized for A40 server deployment

# Build and test on server
server-setup:
	docker-compose build
	docker-compose run --rm chess-dqn-training

# Run Phase 1 tests with GPU monitoring
server-test:
	docker-compose run --rm chess-dqn-training bash -c "nvidia-smi && python test_dqn_phase1.py"

# Start interactive session on server
server-interactive:
	docker-compose run --rm chess-dqn-interactive bash -c "nvidia-smi && /bin/bash"

# Monitor GPU usage during training
gpu-monitor:
	docker-compose exec chess-dqn-training watch -n 1 nvidia-smi

# ============ GPU SELECTION ============
# Use specific GPU (0-9)

# Test on specific GPU
test-gpu:
	@if [ -z "$(GPU)" ]; then echo "Usage: make test-gpu GPU=1"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker-compose run --rm chess-dqn-training python test_dqn_phase1.py

# Interactive session on specific GPU
interactive-gpu:
	@if [ -z "$(GPU)" ]; then echo "Usage: make interactive-gpu GPU=1"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker-compose run --rm chess-dqn-interactive bash -c "nvidia-smi && /bin/bash"

# Training on specific GPU
train-gpu:
	@if [ -z "$(GPU)" ]; then echo "Usage: make train-gpu GPU=1"; exit 1; fi
	CUDA_VISIBLE_DEVICES=$(GPU) docker-compose run --rm chess-dqn-training python src/training/train_dqn.py

# Check GPU status
gpu-status:
	nvidia-smi

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

# API backend
backend:
	docker-compose up backend

# Frontend + Backend
web:
	docker-compose up backend frontend

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
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# Help
help:
	@echo "Chess-RL Training Commands:"
	@echo ""
	@echo "🐳 DOCKER COMMANDS:"
	@echo "  make build          - Build Docker image"
	@echo "  make test-phase1    - Test Phase 1 DQN implementation"
	@echo "  make interactive    - Start interactive container"
	@echo "  make tensorboard    - Start TensorBoard"
	@echo "  make train-dqn      - Start DQN training"
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
	@echo "🧹 UTILITIES:"
	@echo "  make clean          - Clean Python cache"
	@echo "  make clean-docker   - Clean Docker containers"
	@echo "  make help           - Show this help" 
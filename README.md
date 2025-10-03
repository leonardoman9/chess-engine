# Chess Engine with Deep Reinforcement Learning

A chess engine implementation using Deep Q-Networks (DQN) with Convolutional Neural Networks, trained through self-play reinforcement learning to achieve competitive performance against Stockfish.

## Project Objectives

- Target ELO rating: 1100+ against Stockfish depth=1
- Architecture: Dueling DQN with CNN backbone
- Training methodology: Self-play reinforcement learning
- Future extensions: Graph Neural Networks (GNN) with equivariance

## Technical Features

- Deep Q-Network implementation with dueling architecture and CNN layers
- Self-play training environment with experience replay
- Stockfish integration for evaluation and training opponents
- Comprehensive training metrics and game quality analysis
- Docker containerization with GPU support for NVIDIA A40
- TensorBoard integration for training visualization
- Multiple experiment configurations for systematic evaluation
- PGN export and chess game analysis tools

## Prerequisites

- Python 3.10+ (for local development)
- Docker and Docker Compose (recommended deployment method)
- NVIDIA GPU (optional, for accelerated training)
- Stockfish 11+ (included in Docker image)

## Quick Start

### Setup and Build
```bash
git clone <repository-url>
cd chess-engine
make build
```

### Testing Implementation
```bash
# Test Phase 1 DQN implementation
make test-phase1

# Test training setup
make test-training

# Quick training test (100 games)
make train-quick
```

### Results Analysis
```bash
# Analyze training results
make analyze-training RESULTS=results/experiment_dir

# Generate training plots
make training-plots RESULTS=results/experiment_dir

# Generate sample games
make sample-games RESULTS=results/experiment_dir
```

## Training Commands

### Basic Training Configurations
```bash
# List available experiments
make list-experiments

# Quick training (100 games, ~2 minutes)
make train-quick

# Development training (500 games, ~10 minutes)  
make train-dev

# Production training (2000 games, ~40 minutes)
make train-prod
```

### GPU Training (Server Deployment)
```bash
# Setup on server with GPU
make server-setup

# Train on specific GPU
make train-gpu GPU=1

# Monitor GPU usage
make gpu-monitor
```

### Local Development (CPU Only)
```bash
# Test on local machine
make test-mac

# Interactive session
make interactive-mac
```

## Analysis and Monitoring

### Training Analysis
```bash
# Comprehensive training analysis
make analyze-training RESULTS=results/baseline_small_20251003_105305

# Generate training plots
make training-plots RESULTS=results/baseline_small_20251003_105305

# Analyze game quality metrics
make analyze-games
```

### Sample Game Generation
```bash
# Generate sample games from final model
make sample-games RESULTS=results/experiment_dir

# Generate games from specific checkpoint
make sample-games RESULTS=results/experiment_dir CHECKPOINT=checkpoint_game_20.pt
```

### TensorBoard Visualization
```bash
# Start TensorBoard server
make tensorboard

# Access at http://localhost:6006
```

## Container Management

### Basic Operations
```bash
# Build Docker image
make build

# Interactive shell access
make interactive

# View container status
make container-status

# Connect to running container
make container-connect

# View container logs
make container-logs
```

### Service Management
```bash
# Start API backend (port 8000)
make start-api

# Start TensorBoard (port 6006)
make start-tensorboard

# Start Jupyter Lab (port 8888)
make start-jupyter
```

## Project Structure

```
chess-engine/
├── src/                          # Source code
│   ├── models/                   # Neural network architectures
│   │   └── dueling_dqn.py       # Dueling DQN implementation
│   ├── agents/                   # RL agents
│   │   └── dqn_agent.py         # DQN agent with experience replay
│   ├── utils/                    # Utilities
│   │   ├── action_utils.py      # Chess move ↔ action mapping
│   │   └── exploration.py       # Epsilon-greedy exploration
│   ├── replay/                   # Experience replay
│   │   └── replay_buffer.py     # Standard & prioritized replay
│   └── training/                 # Training infrastructure
│       ├── self_play.py         # Self-play trainer
│       └── configs.py           # Experiment configurations
├── results/                      # Training results
│   └── experiment_name_timestamp/
│       ├── checkpoints/         # Model checkpoints
│       ├── logs/               # Training logs
│       ├── sample_games/       # Generated PGN games
│       ├── experiment_info.json # Experiment metadata
│       └── training_history.json # Training metrics
├── api/                         # FastAPI backend
├── frontend/                    # React frontend
├── train_dqn.py                # Main training script
├── test_dqn_phase1.py          # Phase 1 tests
├── analyze_training.py         # Training analysis
├── analyze_games.py            # Game quality analysis
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose (Mac/CPU)
├── docker-compose.server.yml   # GPU server override
├── Makefile                    # Automation commands
└── tasklist.md                 # Project roadmap
```

## Docker Configuration

The project uses Docker containerization with the following configuration:

### Container Services
- `chess-rl`: Main service for training and analysis (container name: `birds-lmannini-e3da-3`)

### Volume Mounts
- `./models`: Model checkpoints and saved weights
- `./checkpoints`: Training checkpoints
- `./logs`: Training and application logs
- `./results`: Experiment results and analysis
- `./data`: Training data and game records

### Environment Variables
- `PYTHONPATH=/app`: Python module path configuration
- `STOCKFISH_PATH=/usr/games/stockfish`: Stockfish executable path
- `CUDA_VISIBLE_DEVICES`: GPU selection for training
- `TORCH_BACKENDS_CUDNN_BENCHMARK=true`: PyTorch optimization

### Port Mappings
- `6006`: TensorBoard visualization
- `8000`: FastAPI backend
- `8888`: Jupyter Lab (optional)

## Experiment Configurations

The system includes predefined experiment configurations:

- `baseline_small`: Quick testing (100 games, 50 moves max)
- `baseline_medium`: Development (500 games, 100 moves max)
- `baseline_large`: Production (2000 games, 200 moves max)
- `server_experiment`: Server training (extended parameters)

Each configuration includes specific settings for:
- Model architecture (CNN layer sizes, hidden dimensions)
- Training parameters (batch size, learning rate, replay buffer size)
- Exploration strategy (epsilon-greedy with decay)
- Evaluation settings (Stockfish depth, time limits)

## Technical Implementation

### Neural Network Architecture
- Dueling DQN with separate value and advantage streams
- CNN backbone for board state processing
- Action space mapping for chess moves (4208 possible actions)
- Experience replay with prioritized sampling support

### Training Process
- Self-play reinforcement learning
- Epsilon-greedy exploration with configurable decay
- Target network with soft updates
- Stockfish integration for evaluation and training opponents

### Analysis Tools
- Training progress visualization
- Game quality assessment
- ELO rating estimation
- PGN game export and analysis 
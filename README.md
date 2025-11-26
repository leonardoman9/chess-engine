# Chess Engine with Deep Reinforcement Learning

A chess engine built with Deep Q-Networks (DQN) and a convolutional dueling head, trained via self-play. The current setup uses a 15-channel board encoding (pieces, turn/castling, attack maps) and a CNN architecture.

## Project Objectives
- Target: 1100+ Elo vs Stockfish depth=1 (long-term, GPU required)
- Architecture: Dueling DQN with CNN backbone
- Method: Self-play RL with reward shaping and Stockfish evaluations

## Current Status (Nov 2025)
- **Stable Training**: Fixed Q-value explosion and non-decreasing loss by implementing reward scaling (0.05x) and correcting input channel mismatch (15 vs 13).
- **Performance**: Training is now stable with Mean Q-values around 0.8-1.0 and Loss around 0.2. Win rate is starting to emerge (~10% after 80 games).
- **Pipeline**: Fully functional Dockerized pipeline for CPU/GPU training.

## Prerequisites
- Python 3.10+ (for local dev)
- Docker and Docker Compose (recommended)
- NVIDIA GPU optional for faster training
- Stockfish 11+ (bundled in Docker image)

## Quick Start
```bash
git clone <repository-url>
cd chess-engine
make build
```

First training (CPU, Docker):
```bash
make train-hydra-small           # 200 games smoke test
make tensorboard                 # view logs at http://localhost:6006
```

## Training Commands
Docker / Hydra:
```bash
make train-hydra-small          # 200 games (Smoke Test)
make train-hydra-medium         # 500 games
make train-hydra-large          # 1000 games
```

**Long Training (Recommended for Results):**
```bash
# 20k games, large model, checkpoints every 7.5k
make train-hydra-custom PARAMS='experiment=baseline_long_cpu model=large experiment.total_games=20000 training.checkpoint_frequency=7500'
```

GPU:
```bash
make train-hydra-gpu GPU=0 EXP=baseline_large PARAMS='experiment.total_games=2000'
make train-hydra-gpu GPU=0 EXP=cnn_gpu_long
```

## Analysis and Monitoring
```bash
# Training analysis and plots
make analyze-training RESULTS=results/<run_dir> --plots

# ELO
make evaluate-elo-quick CHECKPOINT=results/<run_dir>/final_model.pt
make evaluate-elo-full  CHECKPOINT=results/<run_dir>/final_model.pt

# TensorBoard
make tensorboard                 # reads logs/<run>
```

## Suggested Workflow
1. **Smoke test**: `baseline_small` (200 games) to verify the pipeline.
2. **Long run**: Use `make train-hydra-custom` with `baseline_long_cpu` (20k games) or `cnn_gpu_long` if you have a GPU.
3. **Monitor**: Use TensorBoard to watch `mean_q_value` (should be ~0.8-1.2) and `loss` (should be ~0.2).
4. **Evaluate**: After training, run `analyze_training.py` and `evaluate_elo.py` to measure performance.

## Project Structure
```
src/            # core code
  agents/       # dqn_agent.py (DQN logic, epsilon-greedy)
  models/       # dueling_dqn.py (CNN backbone, 15 input channels)
  training/     # self_play.py (Reward shaping, training loop)
  utils/        # action_utils.py, exploration.py
conf/           # Hydra configs
  model/        # Model architectures (small, medium, large)
  experiment/   # Experiment profiles
results/        # Run outputs (checkpoints, logs)
logs/           # TensorBoard logs
Makefile        # Helper commands
train_hydra.py  # Main entrypoint
```

## Key Configuration Notes
- **Reward Scaling**: Rewards are scaled by 0.05 in `self_play.py` to keep Q-values stable.
- **Input Channels**: The model expects 15 input channels (12 pieces + metadata + attack maps). Ensure `model.input_channels=15` in configs.
- **Docker**: The Docker image must be rebuilt (`make build`) if you change code in `src/`, as the source is copied into the image (unless using a dev volume mount).

## Troubleshooting
- **Exploding Q-values**: Check if reward scaling is active.
- **Channel Mismatch Error**: Ensure `conf/model/*.yaml` has `input_channels: 15`.
- **Docker Code Not Updating**: Run `docker compose build --no-cache chess-rl`.

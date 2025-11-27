# Chess Engine with Deep Reinforcement Learning

A chess engine built with Deep Q-Networks (DQN) and a convolutional dueling head, trained via self-play. The current setup uses a 15-channel board encoding (pieces, turn/castling, attack maps) and a CNN architecture.

## Project Objectives
- Target: 1100+ Elo vs Stockfish depth=1 (long-term, GPU required)
- Architecture: Dueling DQN with CNN backbone
- Method: Self-play RL with reward shaping and Stockfish evaluations

## Current Status (Nov 2025)
- **Stability**: Correct 15 input channels, toned-down reward shaping, prioritized replay, and earlier training start (min buffer 5k).
- **GPU path**: Large CNN (128-256-512, hidden 1024) + batch 256 + PER + max_moves 100, checkpoints every 20k for long runs (100k games).
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
# 20k games, large model, final checkpoint at 20k
make train-hydra-custom PARAMS='experiment=baseline_long_cpu model=large experiment.total_games=20000 training.checkpoint_frequency=20000'
```

GPU:
```bash
# 100k games, large CNN, PER, batch 256, checkpoints every 20k
make train-hydra-gpu GPU=0 EXP=cnn_gpu_long PARAMS='training.checkpoint_frequency=20000 agent.min_buffer_size=5000'
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
2. **Long run**: `cnn_gpu_long` (100k games, GPU) or `baseline_long_cpu` (20k, CPU).
3. **Monitor**: TensorBoard; loss in discesa, mean_Q che non esplode (>~3).
4. **Evaluate**: Dopo ogni checkpoint (20k) con `evaluate_elo.py --quick`; a fine run `analyze_training.py --plots`.

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
- **Reward shaping (light)**: catture 0.08×valore, check +0.02, delta materiale 0.01×Δmat, piccoli bonus posizionali (centro/controllo/arrocco/minacce), patta/timeout -4, mate ±20.
- **Input Channels**: 15 (12 pezzi + meta + attack maps). Verificare `input_channels: 15` nei config.
- **Replay / PER**: per run grandi: `replay_type=prioritized`, `batch_size=256`, `buffer_size=250k`, `min_buffer_size=5000`.
- **Long-run training**: `max_moves=100`, `checkpoint_frequency=20000`.
- **Docker**: Ricostruire l’immagine (`make build` o `docker compose build --no-cache chess-rl`) dopo modifiche a `src/` se non si usa il volume dev.

## Troubleshooting
- **Exploding Q-values**: Check if reward scaling is active.
- **Channel Mismatch Error**: Ensure `conf/model/*.yaml` has `input_channels: 15`.
- **Docker Code Not Updating**: Run `docker compose build --no-cache chess-rl`.

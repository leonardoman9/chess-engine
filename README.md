# Chess Engine with Deep Reinforcement Learning

A chess engine implementation using Deep Q-Networks (DQN) with Convolutional Neural Networks, trained through self-play reinforcement learning to achieve competitive performance against Stockfish.

## Project Objectives

- Target ELO rating: 1100+ against Stockfish depth=1
- Architecture: Dueling DQN with CNN backbone
- Training methodology: Self-play reinforcement learning
- Future extensions: Graph Neural Networks (GNN) with equivariance

## Technical Features

- Dueling Deep Q-Network agent with action masking, target network, and experience replay
- Self-play training loop with shaped rewards and optional Stockfish evaluation
- Hydra configuration system with reusable experiment/model/exploration presets
- Docker Compose environment (CPU) plus GPU override for NVIDIA runtimes
- Analysis toolkit for results, PGN generation, and ELO estimation
- Experimental GNN-based DQN architecture and chess graph utilities (phase 2+ roadmap)

> **Stato attuale (apr 2025):** Tutti i componenti della fase 1 sono implementati. Le ottimizzazioni delle fasi 2-3 (mixed precision, parallel environment, logging avanzato, pipeline GNN completa) sono pianificate ma non ancora consolidate.

## Project Status (ott 2025)

- Pipeline RL completa (self-play DQN dueling, replay buffer, target network, reward shaping e valutazioni Stockfish) operativa sia con backbone **CNN** sia **GNN** (Torch Geometric).
- Run dimostrativi completati su CPU (100–2000 game) e su GPU (fino a 10k game). Con risorse limitate l’ELO resta basso (~250), ma la pipeline è pronta per training estesi su hardware più potente.
- Prossimi step per il project work da 3 CFU:
  1. Eseguire run brevi vs run prolungati (CNN) e un test GNN, salvando log/TensorBoard/ELO.
  2. Preparare relazione tecnica, presentazione e repository con istruzioni per la riproducibilità.
  3. Documentare nelle conclusioni che run davvero competitivi richiedono milioni di partite su GPU.

## Prerequisites

- Python 3.10+ (for local development)
- Docker and Docker Compose (recommended deployment method)
- NVIDIA GPU (optional, for accelerated training)
- Stockfish 11+ (included in Docker image)

## Quick Start

### Setup
```bash
git clone <repository-url>
cd chess-engine
make install        # facoltativo, per esecuzione locale
make build          # facoltativo, per utilizzare Docker
```

### Primo allenamento (CPU locale)
```bash
python train_hydra.py experiment=baseline_small device=cpu
```

### Primo allenamento (Docker)
```bash
make train-hydra-small
```

### Analisi rapida dei risultati
```bash
# Supponendo che Hydra salvi in results/<run_dir>
make analyze-training RESULTS=results/<run_dir>
make evaluate-elo-quick CHECKPOINT=results/<run_dir>/final_model.pt

# Visualizza i log TensorBoard (eventi scritti automaticamente in logs/<run_dir>)
make tensorboard
```

## Training Commands

### Hydra (Docker)
```bash
make train-hydra-small        # 100 partite (~smoke test)
make train-hydra-medium       # 500 partite
make train-hydra-large        # 1000 partite
make train-hydra-custom PARAMS='experiment=baseline_large experiment.total_games=2000 device=cpu'
make train-hydra-custom PARAMS='experiment=cnn_gpu_long device=cpu'   # 10k game (CNN) anche su CPU (molto lento)
```

### GPU Training
```bash
make train-hydra-gpu GPU=0 EXP=baseline_large PARAMS='experiment.total_games=2000'
make train-hydra-gpu GPU=0 EXP=cnn_gpu_long
make train-hydra-gpu GPU=0 EXP=gnn_gpu_long
```

### Sviluppo locale (senza Docker)
```bash
make train-local EXP=baseline_small
# oppure
python train_hydra.py experiment=baseline_medium device=cpu
```

### Direct Hydra Usage
```bash
# Custom experiment configuration
python train_hydra.py experiment=baseline_small model=medium device=cpu

# Override specific parameters
python train_hydra.py experiment=baseline_large experiment.total_games=2000 model.conv_channels=128

# Multi-run parameter sweep
python train_hydra.py --multirun experiment=baseline_small,baseline_medium model=small,medium
```

## Analysis and Monitoring

### Training Analysis
```bash
# Comprehensive training analysis
make analyze-training RESULTS=results/baseline_small_20251003_105305

# Analyze game quality metrics
make analyze-games RESULTS=results/experiment_dir
```

### ELO Evaluation
```bash
# Quick ELO evaluation (3 Stockfish levels)
make evaluate-elo-quick CHECKPOINT=results/experiment_dir/final_model.pt

# Full ELO evaluation (6 Stockfish levels)
make evaluate-elo-full CHECKPOINT=results/experiment_dir/final_model.pt

# Evaluate any model checkpoint
python evaluate_elo.py results/experiment_dir/checkpoints/checkpoint_game_50.pt
```

### Sample Game Generation
```bash
# Generate sample games from a checkpoint
python analyze_training.py results/experiment_dir --generate-games --checkpoint checkpoints/checkpoint_game_20.pt --games 5
```

### TensorBoard Visualization
```bash
# Start TensorBoard server
make tensorboard

# Access at http://localhost:6006

# Per aprire una run specifica
make tensorboard LOGDIR=/app/logs/<nome_run>
```

## Suggested Workflow for High-ELO Runs

1. **Calibrazione CNN** – Avvia `baseline_large` (2000 game) o direttamente `cnn_gpu_long` su GPU. Monitora loss, epsilon e valutazioni Stockfish.
2. **Sessione estesa** – Se l'ELO resta basso, esegui `cnn_gpu_long` (10 000 game) per consolidare la policy. Su CPU richiede molte ore; su GPU ~45‑60 min.
3. **Passaggio GNN** – Quando la CNN si stabilizza, lancia `gnn_gpu_long` (10 000 game) per sfruttare la rappresentazione a grafo. Confronta le run in TensorBoard selezionando i log corrispondenti.
4. **Valutazione periodica** – Dopo ogni run esegui:
   ```bash
   docker compose run --rm chess-rl python analyze_training.py results/<run_dir> --plots
   make evaluate-elo-full CHECKPOINT=results/<run_dir>/final_model.pt
   ```
5. **Iterazione** – Regola `epsilon_decay_steps`, `stockfish_depth/time` e profilo `training` in base ai risultati. L'obiettivo 1100+ Elo richiede frequenti run da 5‑10k partite e, se possibile, l'uso di GPU.
6. **Consegna accademica** – Salva grafici, stime Elo e PGN significativi per alimentare relazione tecnica, presentazione e repository da consegnare per l'esame.

## Container Management

### Basic Operations
```bash
# Build Docker image
make build

# Interactive shell access
make shell
```

### Development Services
```bash
# Start TensorBoard (port 6006)
make tensorboard

# Start Jupyter Lab (port 8888) 
make notebook
```

## Project Structure

```
chess-engine/
├── 📁 src/                       # Core source code modules
│   ├── 📁 agents/               # Reinforcement learning agents
│   │   └── dqn_agent.py         # DQN agent with experience replay and target network
│   ├── 📁 models/               # Neural network architectures
│   │   └── dueling_dqn.py       # Dueling DQN implementation with CNN backbone
│   ├── 📁 replay/               # Experience replay systems
│   │   └── replay_buffer.py     # Standard and prioritized experience replay buffers
│   ├── 📁 training/             # Training infrastructure
│   │   ├── self_play.py         # Self-play trainer with Stockfish integration
│   │   └── configs.py           # Predefined experiment configurations
│   └── 📁 utils/                # Utility modules
│       ├── action_utils.py      # Chess move ↔ action index mapping (4208 actions)
│       ├── exploration.py       # Epsilon-greedy exploration strategies
│       ├── elo_calculator.py    # ELO rating evaluation against multiple Stockfish levels
│       └── game_generator.py    # PGN game generation and analysis tools
├── 📁 conf/                     # Hydra configuration files
│   ├── config.yaml             # Main configuration entry point
│   ├── 📁 experiment/          # Experiment presets (baseline, extended, server, GNN, AlphaZero)
│   ├── 📁 model/               # Model configs (CNN small/medium/large + GNN/AlphaZero variants)
│   └── 📁 training/            # Training parameter sets (quick, improved_quick, development, production, server)
├── 📁 results/                  # Training experiment results
│   └── experiment_name_timestamp/  # Individual experiment directory
│       ├── 📁 checkpoints/     # Model checkpoints during training
│       ├── 📁 sample_games/    # Generated PGN games (self-play and vs Stockfish)
│       ├── 📁 plots/           # Training visualization plots
│       ├── 📁 elo_evaluation/  # ELO evaluation results and analysis
│       ├── experiment_info.json # Experiment metadata and configuration
│       ├── training_history.json # Episode rewards, lengths, and evaluation metrics
│       ├── final_model.pt      # Final trained model weights
│       └── final_model_buffer.pt # Final experience replay buffer state
├── 📁 logs/                     # TensorBoard and application logs
├── 📄 train_hydra.py           # Main training script with Hydra configuration support
├── 📄 evaluate_elo.py          # Standalone ELO evaluation script for any model
├── 📄 analyze_training.py      # Comprehensive training results analysis
├── 📄 analyze_games.py         # Chess game quality analysis and metrics
├── 📄 requirements.txt         # Python dependencies (PyTorch, chess, stockfish, etc.)
├── 📄 Dockerfile              # Docker container configuration for training
├── 📄 docker-compose.yml      # Docker Compose for local development (CPU)
├── 📄 docker-compose.server.yml # Docker Compose override for GPU server deployment
├── 📄 Makefile                # Helper targets for build, training, analysis, and services
├── 📄 README.md               # Project documentation (this file)
└── 📄 tasklist.md             # Detailed project roadmap and implementation phases
```

### 📄 Core Files Description

#### Training and Execution
- **`train_hydra.py`**: Main training script using Hydra for flexible configuration management. Handles self-play training, model checkpointing, ELO evaluation, sample game generation, and automatic plotting.
- **`evaluate_elo.py`**: Standalone script for evaluating any trained model's ELO rating against multiple Stockfish difficulty levels (400-1600 ELO). Automatically detects model configuration from checkpoints.

#### Analysis and Monitoring  
- **`analyze_training.py`**: Comprehensive analysis tool for training results. Generates plots, statistics, and insights from training history and checkpoints.
- **`analyze_games.py`**: Chess-specific game quality analysis. Evaluates move validity, game outcomes, tactical patterns, and playing strength indicators.

#### Configuration and Deployment
- **`requirements.txt`**: Python package dependencies including PyTorch, python-chess, stockfish, matplotlib, seaborn, and Hydra.
- **`Dockerfile`**: Container definition with PyTorch CUDA support, Stockfish installation, and optimized training environment.
- **`docker-compose.yml`**: Local development setup with CPU-only training and volume mounts for results persistence.
- **`docker-compose.server.yml`**: GPU server deployment override with NVIDIA runtime and GPU resource allocation.
- **`Makefile`**: Helper targets for local/Docker training, analysis, evaluation, and developer services.

#### Documentation and Planning
- **`README.md`**: Complete project documentation with usage examples, configuration options, and technical specifications.
- **`tasklist.md`**: Detailed project roadmap with 8 implementation phases, from core DQN to advanced GNN architectures.

### 📁 Source Code Modules

#### `src/agents/`
- **`dqn_agent.py`**: Complete DQN agent implementation with experience replay, target network updates, epsilon-greedy exploration, and action masking for legal chess moves.

#### `src/models/`  
- **`dueling_dqn.py`**: Dueling DQN architecture with CNN backbone for chess board processing. Includes model factory functions and configuration management for different network sizes.

#### `src/replay/`
- **`replay_buffer.py`**: Experience replay buffer implementations including standard uniform sampling and prioritized experience replay with importance sampling.

#### `src/training/`
- **`self_play.py`**: Self-play training orchestrator managing game generation, model updates, evaluation against Stockfish, and training metrics collection.
- **`configs.py`**: Predefined experiment configurations with different complexity levels for systematic evaluation and development.

#### `src/utils/`
- **`action_utils.py`**: Chess move encoding/decoding utilities mapping between UCI chess moves and neural network action indices (4208 total actions).
- **`exploration.py`**: Epsilon-greedy exploration strategies with linear, exponential, and cosine decay schedules for training optimization.
- **`elo_calculator.py`**: ELO rating calculation system for comprehensive model evaluation against multiple Stockfish configurations with statistical confidence intervals.
- **`game_generator.py`**: PGN game generation utilities for creating sample games, self-play demonstrations, and training analysis datasets.

### 📁 Configuration System (Hydra)

The project uses Hydra for flexible, reproducible experiment configuration:

- **`conf/config.yaml`**: Main configuration entry point defining default experiment, model, and training parameters
- **`conf/experiment/`**: Experiment-specific configurations for different training scenarios (quick testing to intensive server training)
- **`conf/model/`**: Neural network architecture configurations (small/medium/large CNN variants)
- **`conf/training/`**: Training hyperparameter sets optimized for different computational budgets

### 📁 Results Organization

Each training run creates a timestamped directory containing:
- **Model artifacts**: Final weights, checkpoints, and replay buffer states
- **Analysis outputs**: Training plots, ELO evaluations, and sample games
- **Metadata**: Complete experiment configuration and training history for reproducibility

## Docker Configuration

The project uses Docker containerization optimized for chess engine training:

### Container Services
- **`chess-rl`**: Main training and analysis service (container name: `birds-lmannini-e3da-3`)

### Volume Mounts (Optimized)
- `./logs`: TensorBoard logs and training output
- `./results`: Experiment results, models, and analysis artifacts

### Environment Variables
- `PYTHONPATH=/app`: Python module path configuration
- `STOCKFISH_PATH=/usr/games/stockfish`: Stockfish executable path  
- `CUDA_VISIBLE_DEVICES`: GPU selection for training (server deployment)
- `TORCH_BACKENDS_CUDNN_BENCHMARK=true`: PyTorch CUDA optimization
- `NVIDIA_VISIBLE_DEVICES=all`: NVIDIA GPU visibility (server only)

### Port Mappings
- `6006`: TensorBoard visualization interface
- `8888`: Jupyter Lab (optional development environment)

### Deployment Modes
- **Local Development** (`docker-compose.yml`): CPU-only training for testing and development
- **GPU Server** (`docker-compose.server.yml`): NVIDIA GPU-accelerated training with runtime optimization

## Experiment Configurations

The system uses Hydra for flexible experiment configuration with predefined setups:

### Training Experiments *(conf/experiment/)*
- **`baseline_small`** – 100 games · `training=quick` (100 mosse, valutazioni frequenti) · `agent=small`. Pensato come smoke test (<5 min).
- **`baseline_medium`** – 500 games · `training=development` · `agent=default`. Profilo di sviluppo standard.
- **`baseline_large`** – 2000 games · `training=production` · `agent=large`. Run “production” (CNN large, buffer prioritizzato).
- **`cnn_gpu_long`** – 10 000 games · CNN large · `training=production_long` · `exploration=balanced_fast`. Pensato per GPU; run esteso per scalare l’ELO.
- **`gnn_gpu_long`** – 10 000 games · `model=gnn_pro_a40` · stesso profilo GPU; richiede torch-geometric e fornisce la pipeline grafo completa.
- **`extended_training`**, **`server_intensive`** – 3000‑5000 games con profili `production/server`. Usali quando vuoi massimizzare l’ELO.
- **`gnn_*`** (baseline/medium/pro) – Configurazioni che istanziano `src.models.gnn_dqn.GNNDQN` con diverse architetture. Richiedono Torch Geometric.

### Model Architectures *(conf/model/)*
- **`small` / `medium` / `large`** – CNN dueling con rispettivamente [32,64,128], [64,128,256], [128,256,512] canali e hidden 256/512/1024.
- **`improved_*`, `large_kernel`, `alphazero_*`** – Varianti sperimentali della CNN.
- **`gnn_*`** – Modelli grafo (ResGATEAU, GAT/GINE, multiscale). Ogni file definisce `_target_: src.models.gnn_dqn.GNNDQN` con parametri di nodi/edge/pooling.

### Training Profiles *(conf/training/)*
- **`quick`** – 100 mosse, 5 game/episodio, update ogni 2 game: utile per test rapidi.
- **`improved_quick`** – 100 mosse, 10 game/episodio, timeout 30s: versione “rapida” più realistica.
- **`development`** – 100 mosse, 10 game/episodio, valutazione ogni 25 game: profilo default.
- **`production`** – 100 mosse, 20 game/episodio, timeout 60s, eval/games 100/20: run prolungati su CPU/GPU.
- **`production_long`** – 120 mosse, 25 game/episodio, valutazione ogni 200 game, timeout 90s: run lunghi da 10k partite.
- **`server`** – 100 mosse, 50 game/episodio, depth 2, timeout 120s: pensato per GPU con molta memoria.

Each configuration automatically includes:
- **Model checkpointing**: Periodic saves during training
- **ELO evaluation**: Multi-level Stockfish assessment (400-1600 ELO) with graceful fallback
- **Sample games**: PGN generation for analysis
- **Training plots**: Automatic visualization of progress
- **Experiment tracking**: Complete reproducibility metadata

## Technical Implementation

### Neural Network Architecture
- **Dueling DQN**: Separate value and advantage streams for improved learning stability
- **CNN Backbone**: Convolutional layers for spatial chess board feature extraction
- **Action Space**: 4208 possible chess moves encoded as action indices
- **Experience Replay**: Standard and prioritized sampling with importance weighting
- **Target Network**: Soft updates for training stability (τ = 0.005)

### Training Process
- **Self-Play**: Agent plays against itself with epsilon-greedy exploration
- **Stockfish Integration**: Multi-level evaluation opponents (400-1600 ELO)
- **Exploration Strategies**: Linear, exponential, and cosine epsilon decay
- **Action Masking**: Legal move filtering for chess rule compliance
- **Batch Learning**: Configurable batch sizes with replay buffer sampling

### Evaluation System
- **Multi-Level ELO**: Comprehensive rating against 6 Stockfish difficulty levels
- **Statistical Analysis**: Confidence intervals and performance metrics
- **Game Quality**: Move validity, tactical pattern, and outcome analysis
- **Progress Tracking**: Training curves, loss monitoring, and convergence analysis

### Analysis and Visualization
- **Training Plots**: Reward progression, episode length, epsilon decay, loss curves
- **ELO Evaluation**: Rating estimation with confidence intervals and opponent analysis
- **Sample Games**: PGN export for human analysis and pattern recognition
- **Performance Metrics**: Win/draw/loss rates, game length statistics, move quality assessment 

### Hydra Overrides & Key Parameters
- **Selezione profili**: `python train_hydra.py experiment=baseline_large exploration=balanced agent=default training=production`.
- **Parametri principali**
  - `total_games` – numero totale di partite generate.
  - `games_per_episode` – partite giocate prima di loggare un “episode”.
  - `training_frequency` – ogni quanti game viene chiamato `train_step`.
  - `target_update_frequency` – frequenza di soft update del target network.
  - `eval_frequency` / `eval_games` – partite contro Stockfish usate per monitorare i progressi.
  - `max_moves` / `game_timeout` – limiti per interrompere game lunghi (timeout penalizzato).
  - `stockfish_depth` / `stockfish_time` – forza dell’avversario durante training/eval.
  - `epsilon_start`, `epsilon_end`, `epsilon_decay_steps` – controllano il decadimento ε-greedy. Con `strategy_type=linear` si usa:  
    ε(t) = ε_start + (ε_end − ε_start) · min(t/decay_steps, 1).  
    Esempio in `balanced.yaml`: 200 k step per scendere da 1.0 a 0.1; `balanced_fast.yaml` usa 50 k step e `epsilon_end=0.05` per run aggressivi.
  - `buffer_size`, `min_buffer_size` – capacità del replay buffer e warmup minimo prima di addestrare.
  - `batch_size`, `learning_rate`, `gamma`, `tau` – iperparametri standard di DQN (sconto, soft update).
  - `replay_type` – `standard` o `prioritized` (usa TD-error e importance sampling).
- **Esempi utili**
  - CNN produzione: `make train-hydra-large` (usa produzione + agent large).
  - Run esteso su GPU: `make train-hydra-gpu GPU=0 EXP=cnn_gpu_long`.
  - GNN baseline: `make train-hydra-custom PARAMS='experiment=gnn_baseline device=cpu'`.
  - GNN lungo su GPU: `make train-hydra-gpu GPU=0 EXP=gnn_gpu_long`.
- **Suggerimento**: dopo il warmup iniziale, monitora `exploration/epsilon_episode`. Se ε resta >0.9 per gran parte del run, riduci `exploration.epsilon_decay_steps` (es. 5000) e/o aumenta `training.training_frequency` per aumentare il numero di update per partita.

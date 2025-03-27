# Chess Engine with Neural Networks

A chess engine that uses neural networks trained through self-play to evaluate positions and make moves.

## Features

- Neural network-based position evaluation
- Self-play training with MCTS (Monte Carlo Tree Search)
- GUI interface for playing against the engine
- ELO rating evaluation against Stockfish
- PGN game saving and analysis
- Docker support for easy deployment

## Prerequisites

- Python 3.9 or higher
- Poetry for dependency management
- Stockfish chess engine
- Docker and Docker Compose (optional, for containerized deployment)

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chess-engine
```

2. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install
```

4. Install Stockfish:
- On macOS: `brew install stockfish`
- On Ubuntu/Debian: `sudo apt-get install stockfish`
- On Windows: Download from [Stockfish website](https://stockfishchess.org/download/)

### Docker Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chess-engine
```

2. Build the Docker image:
```bash
docker-compose build
```

## Usage

### Local Usage

1. Training the model:
```bash
poetry run python train.py
```

2. Playing against the model:
```bash
poetry run python test_model.py
```

3. Evaluating ELO rating:
```bash
poetry run python evaluate_elo.py --model content/models/checkpoint_10.pt --stockfish /path/to/stockfish --time 1.0 --games 10 --gui true
```

### Docker Usage

1. Training the model:
```bash
docker-compose run --rm chess-engine poetry run python train.py
```

2. Playing against the model:
```bash
docker-compose run --rm chess-engine poetry run python test_model.py
```

3. Evaluating ELO rating:
```bash
docker-compose run --rm chess-engine poetry run python evaluate_elo.py --model content/models/checkpoint_10.pt --stockfish /usr/games/stockfish --time 1.0 --games 10 --gui true
```

4. Saving games as PGN:
```bash
docker-compose run --rm chess-engine poetry run python evaluate_elo.py --model content/models/checkpoint_10.pt --stockfish /usr/games/stockfish --time 1.0 --games 10 --save
```

## Project Structure

```
chess-engine/
├── data/                  # Training and evaluation data
├── models/               # Saved model checkpoints
├── evaluation_results/   # ELO evaluation results
├── train.py             # Training script
├── test_model.py        # GUI for playing against the model
├── evaluate_elo.py      # ELO rating evaluation script
├── gui.py              # GUI implementation
├── pyproject.toml      # Poetry dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
└── .dockerignore      # Docker ignore rules
```

## Docker Configuration

The project includes Docker support with the following services:

- `chess-engine`: Main service for running the chess engine
- `training`: Service for model training
- `interface`: Web interface service (port 8000)
- `tensorboard`: TensorBoard visualization service (port 6006)

### Docker Volumes

The following directories are mounted as volumes:
- `./data`: Training and evaluation data
- `./models`: Saved model checkpoints
- `./evaluation_results`: ELO evaluation results

### Environment Variables

- `PYTHONPATH`: Set to `/app` for proper module imports
- `STOCKFISH_PATH`: Path to Stockfish executable in the container
- `NUM_WORKERS`: Number of training workers (default: 4)
- `BATCH_SIZE`: Training batch size (default: 512)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
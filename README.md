# Chess Engine with Neural Network

This project implements a chess engine using a neural network trained through self-play. The engine is evaluated against Stockfish with configurable strength levels.

## Project Structure

```
.
├── models/                  # Directory for saved model checkpoints
├── test_model.py           # GUI interface for playing against the model
├── evaluate_model.py       # Script to evaluate model against Stockfish
├── train_on_colab.ipynb    # Google Colab notebook for training
└── README.md              # Project documentation
```

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install torch numpy chess python-chess tqdm PyQt5
```
3. Install Stockfish:
   - macOS: `brew install stockfish`
   - Linux: `sudo apt-get install stockfish`
   - Windows: Download from [Stockfish website](https://stockfishchess.org/download/)

## Usage

### Playing Against the Model

To play against the model using the GUI interface:

```bash
python test_model.py --model content/models/checkpoint_10.pt
```

### Evaluating Model Strength

To evaluate the model's strength against Stockfish:

```bash
python evaluate_elo.py --model content/models/checkpoint_10.pt --stockfish /opt/homebrew/bin/stockfish --time 1.0 --games 10 --stockfish-elo 1500 --save
```

#### Command Line Arguments

- `--model`: Path to the model checkpoint (required)
- `--stockfish`: Path to Stockfish executable (required)
- `--time`: Time control in seconds per move (default: 1.0)
- `--games`: Number of games to play (default: 10)
- `--stockfish-elo`: Stockfish ELO rating (100-3190, default: 3000)
- `--gui`: Show GUI during evaluation
- `--save`: Save games as PGN files

#### Stockfish Strength Configuration

The script supports two modes for configuring Stockfish's strength:

1. **UCI_Elo Mode** (1320-3190 ELO):
   - Used when `--stockfish-elo` is 1320 or higher
   - Provides precise ELO-based strength limiting
   - Example: `--stockfish-elo 1500`

2. **Skill Level Mode** (100-1320 ELO):
   - Used when `--stockfish-elo` is below 1320
   - Maps ELO ratings to Stockfish's internal skill levels (0-20)
   - Formula: `skill_level = (elo - 100) / 61`
   - Example: `--stockfish-elo 500` will use Skill Level 6

#### PGN File Saving

When using the `--save` flag, games are saved in the following format:
```
data/eval_<date>_<modelname>_<stockfishelo>/game_<number>.pgn
```

Each PGN file includes:
- Complete game moves
- Game metadata (players, date, ELO ratings)
- Final result

## Model Architecture

The neural network consists of:
- Input layer: 13 channels (6 piece types × 2 colors + empty squares)
- 3 convolutional layers (64, 128, 256 filters)
- 2 fully connected layers (1024, 512 neurons)
- Two output heads:
  - Value head: Evaluates position
  - Policy head: Predicts move probabilities

## Training Process

The model is trained through self-play with the following parameters:
- Learning rate: 0.001
- Batch size: 256
- Number of games: 1000
- Loss functions:
  - Value loss: MSE
  - Policy loss: Cross-entropy

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
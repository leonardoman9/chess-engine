.PHONY: install build run play evaluate clean

# Install dependencies
install:
	poetry install

# Build Docker image
build:
	docker-compose build

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

# Docker commands
docker-play:
	docker-compose run --rm chess-engine poetry run python test_model.py --model content/models/checkpoint_10.pt

docker-evaluate:
	docker-compose run --rm chess-engine poetry run python evaluate_elo.py --model content/models/checkpoint_10.pt --stockfish /usr/games/stockfish --time 1.0 --games 10 --stockfish-elo 3000 --save

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} + 
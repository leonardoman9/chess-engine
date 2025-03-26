.PHONY: build train play test lint format clean

# Build the Docker image
build:
	docker-compose build

# Start training
train:
	docker-compose up training

# Start the interface
play:
	docker-compose up interface

# Start TensorBoard
tensorboard:
	docker-compose up tensorboard

# Run tests
test:
	docker-compose run --rm training pytest

# Run linting
lint:
	docker-compose run --rm training black src tests
	docker-compose run --rm training isort src tests
	docker-compose run --rm training mypy src
	docker-compose run --rm training pylint src tests

# Format code
format:
	docker-compose run --rm training black src tests
	docker-compose run --rm training isort src tests

# Clean up
clean:
	docker-compose down
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} + 
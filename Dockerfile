# Use PyTorch official image with CUDA support for A40
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    tzdata \
    stockfish \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for DQN training
RUN pip install tensorboard matplotlib seaborn

# Copy project files
COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY test_dqn_phase1.py .
COPY test_training.py .
COPY train_dqn.py .
COPY analyze_games.py .
COPY analyze_training.py .
COPY tasklist.md .

# Create directories for training artifacts
RUN mkdir -p logs checkpoints results data/training

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_BACKENDS_CUDNN_BENCHMARK=true

# Default command runs Phase 1 tests
CMD ["python", "test_dqn_phase1.py"] 
# Use PyTorch official image with CUDA support for A40
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

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
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric with correct versions for CPU fallback
RUN pip install --no-cache-dir \
    torch-scatter==2.1.2+pt21cu118 \
    torch-sparse==0.6.18+pt21cu118 \
    torch-cluster==1.6.3+pt21cu118 \
    torch-spline-conv==1.2.2+pt21cu118 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html && \
    pip install --no-cache-dir torch-geometric==2.4.0

# Install additional dependencies for DQN training
RUN pip install tensorboard matplotlib seaborn

# Copy project files
COPY src/ src/
COPY train_hydra.py .
COPY analyze_games.py .
COPY analyze_training.py .
COPY evaluate_elo.py .
COPY tasklist.md .
COPY conf/ conf/

# Create directories for training artifacts
RUN mkdir -p logs results

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_BACKENDS_CUDNN_BENCHMARK=true

# Default command runs Hydra training
CMD ["python", "train_hydra.py"] 

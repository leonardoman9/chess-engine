# Build stage
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Runtime stage
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy source code from builder
COPY --from=builder /app/src/ src/

# Create necessary directories
RUN mkdir -p models data

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.training"] 
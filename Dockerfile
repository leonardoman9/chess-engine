# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy only the files needed for dependency installation
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary project files
COPY api/ api/
COPY models/ models/

# Set the default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 
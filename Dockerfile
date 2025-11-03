# Start from an official Python runtime (slim). Use a stable Python version.
FROM python:3.10-slim

# Metadata
LABEL maintainer="mohamed-em2m"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV (and for many TF wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# If you prefer tensorflow-cpu to reduce size, pin it in requirements.txt (or replace below)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY models ./models
COPY .env . || true

# Create a non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose port used by Uvicorn
EXPOSE 8000

# Default environment variables (can be overridden at runtime)
ENV MODEL_PATH=models/model.h5
ENV CLASS_NAMES=""

# Command to run the FastAPI app with Uvicorn
# Use --host 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
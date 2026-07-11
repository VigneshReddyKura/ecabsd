# Use a lightweight official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Set the working directory in the container
WORKDIR /app

# Create a non-root group and user for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Install system dependencies needed for compilation, downloading files, and PyG
# Clean up apt caches to minimize image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install AutoDock Vina Linux binary
RUN wget -q https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 -O /usr/local/bin/vina \
    && chmod +x /usr/local/bin/vina

# Copy only the requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install PyTorch (CPU-only) and PyG dependencies in correct order to avoid compilation errors
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cpu.html && \
    pip install --no-cache-dir torch-geometric==2.7.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Grant ownership to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Pre-download the best model weights during the build phase
# This prevents downloading weights on every startup and speeds up container launches
RUN python download_weights.py

# Expose the default FastAPI port
EXPOSE 8000

# Add a HEALTHCHECK to verify application is responding
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

# Start the web server
CMD ["python", "web/app.py"]

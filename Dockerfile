# SCOTUS AI Docker Container
# Multi-stage build for optimized image size and better caching
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies in builder stage
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional dependencies that may be needed based on the codebase
RUN pip install --no-cache-dir \
    sentence-transformers \
    loguru \
    spacy \
    datasets \
    seaborn \
    matplotlib \
    jupyter \
    ipykernel

# Download spaCy model
RUN python3 -m spacy download en_core_web_sm

# Production stage
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime system dependencies (NO build tools)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-distutils \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    libjpeg8 \
    libpng16-16 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage (now fully populated)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/data/external \
    /app/logs \
    /app/models_output \
    /app/.cache \
    /app/notebooks

# Copy project files
COPY . /app/

# Create a minimal README.md if it doesn't exist (for setup.py)
RUN echo "# SCOTUS AI" > /app/README.md || true

# Install the project in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH="/app:/app/scripts"
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Default environment variables (can be overridden)
ENV LOG_LEVEL=INFO
ENV LOG_FILE=/app/logs/scotus_ai.log
ENV MODEL_OUTPUT_DIR=/app/models_output/
ENV CACHE_DIR=/app/.cache/
ENV SCRAPER_DELAY=1.0
ENV MAX_RETRIES=3
ENV USER_AGENT="SCOTUS-AI-Bot/1.0"

# Copy entrypoint script and set permissions while still root
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create a non-root user for security
RUN useradd -m -s /bin/bash scotus && \
    chown -R scotus:scotus /app
USER scotus

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Volume mounts for data persistence
VOLUME ["/app/data", "/app/logs", "/app/models_output", "/app/.cache"]

# Expose ports (if needed for future web interface)
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["bash"] 
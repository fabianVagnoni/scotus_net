#!/bin/bash
set -e

# Create required directories if they don't exist
mkdir -p /app/data/raw /app/data/processed /app/data/augmented /app/logs /app/logs/hyperparameter_tunning_logs /app/logs/training_logs /app/models_output /app/.cache

# Set proper permissions only on directories we can modify
# Skip permission changes on mounted volumes to avoid "Operation not permitted" errors
echo "Setting up permissions..."

# Try to make logs directory writable (this is critical for the application)
if [ -d "/app/logs" ]; then
    # Try to create a test file to check if we can write to logs
    if ! touch /app/logs/.write_test 2>/dev/null; then
        echo "WARNING: Cannot write to /app/logs directory. Logging may not work properly."
        echo "Consider running: sudo chown -R $(id -u):$(id -g) logs/ on the host system"
        # Try to use a fallback logs directory
        mkdir -p /tmp/scotus_logs 2>/dev/null || true
        export LOG_FILE="/tmp/scotus_logs/scotus_ai.log"
        echo "Using fallback log directory: /tmp/scotus_logs/"
    else
        rm -f /app/logs/.write_test 2>/dev/null || true
        echo "Logs directory is writable"
    fi
fi

# Only set permissions on directories that are typically not mounted from host
chmod -R 755 /app/models_output /app/.cache 2>/dev/null || echo "Note: Some permission changes skipped (this is normal for mounted volumes)"

# Ensure we can write to data directories by creating them if needed, but don't change existing file permissions
mkdir -p /app/data/raw /app/data/processed /app/data/augmented /app/data/external 2>/dev/null || true

# Check if .env file exists, if not create from template
if [ ! -f /app/.env ] && [ -f /app/env.example ]; then
    echo "Creating .env file from template..."
    cp /app/env.example /app/.env
fi

# Print usage information
echo "SCOTUS AI Docker Container"
echo "=========================="
echo ""
echo "Available commands:"
echo "  data-pipeline      - Run the complete data pipeline"
echo "  data-pipeline-step - Run a specific pipeline step"
echo "  encoding           - Run the encoding pipeline"
echo "  augmentation       - Run the augmentation pipeline"
echo "  train              - Train the model with optimized hyperparameters"
echo "  hyperparameter-tuning - Run hyperparameter optimization"
echo "  check              - Check data status"
echo "  bash               - Open bash shell"
echo ""
echo "Examples:"
echo "  docker run -it scotus-ai data-pipeline"
echo "  docker run -it scotus-ai data-pipeline-step scrape-justices"
echo "  docker run -it scotus-ai encoding"
echo "  docker run -it scotus-ai augmentation"
echo "  docker run -it scotus-ai train --experiment-name production_v1"
echo "  docker run -it scotus-ai hyperparameter-tuning --experiment-name test"
echo "  docker run -it scotus-ai bash"
echo ""

# Execute the command
case "$1" in
    data-pipeline)
        shift
        exec python3 -m scripts.data_pipeline.main "$@"
        ;;
    data-pipeline-step)
        shift
        exec python3 -m scripts.data_pipeline.main --step "$@"
        ;;
    encoding)
        shift
        exec python3 -m scripts.tokenization.main_encoder "$@"
        ;;
    encode-bios)
        shift
        exec python3 -m scripts.tokenization.encode_bios "$@"
        ;;
    encode-descriptions)
        shift
        exec python3 -m scripts.tokenization.encode_descriptions "$@"
        ;;
    augmentation)
        shift
        exec python3 -m augmentation.main "$@"
        ;;
    train)
        shift
        exec python3 -m scripts.models.run_training "$@"
        ;;
    hyperparameter-tuning)
        shift
        exec python3 -m scripts.models.hyperparameter_optimization "$@"
        ;;
    check)
        shift
        exec python3 -m scripts.data_pipeline.main --check "$@"
        ;;
    bash)
        exec /bin/bash
        ;;
    *)
        if [ -z "$1" ]; then
            echo "No command specified. Use 'bash' to open shell or specify a command."
            exec /bin/bash
        else
            echo "Running custom command: $@"
            exec "$@"
        fi
        ;;
esac 
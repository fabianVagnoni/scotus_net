#!/bin/bash
set -e

# Create required directories if they don't exist
mkdir -p /app/data/raw /app/data/processed /app/logs /app/models_output /app/.cache

# Set proper permissions
chmod -R 755 /app/data /app/logs /app/models_output /app/.cache

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
echo "  train              - Train the model"
echo "  check              - Check data status"
echo "  bash               - Open bash shell"
echo ""
echo "Examples:"
echo "  docker run -it scotus-ai data-pipeline"
echo "  docker run -it scotus-ai data-pipeline-step scrape-justices"
echo "  docker run -it scotus-ai encoding"
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
    train)
        shift
        exec python3 -m scripts.models.model_trainer "$@"
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
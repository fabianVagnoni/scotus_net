#!/bin/bash

# SCOTUS AI Docker Management Script
# ==================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if nvidia-docker is available (for GPU support)
check_nvidia_docker() {
    if command -v nvidia-docker >/dev/null 2>&1; then
        print_info "NVIDIA Docker support detected"
        return 0
    elif docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >/dev/null 2>&1; then
        print_info "Docker GPU support detected"
        return 0
    else
        print_warning "NVIDIA Docker support not available. GPU features will be disabled."
        return 1
    fi
}

# Function to create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    mkdir -p data/raw data/processed logs logs/hyperparameter_tunning_logs models_output cache
    print_success "Directories created"
}

# Function to build the Docker image
build_image() {
    print_info "Building SCOTUS AI Docker image..."
    docker build -t scotus-ai:latest .
    print_success "Docker image built successfully"
}

# Function to run the container with proper setup
run_container() {
    local command="$1"
    shift
    local args="$@"
    
    check_docker
    create_directories
    
    # Check if image exists
    if ! docker image inspect scotus-ai:latest >/dev/null 2>&1; then
        print_info "Image not found. Building..."
        build_image
    fi
    
    # Prepare Docker run command
    local docker_cmd="docker run --rm -it"
    
    # Add GPU support if available
    if check_nvidia_docker; then
        docker_cmd="$docker_cmd --gpus all"
    fi
    
    # Add volume mounts
    docker_cmd="$docker_cmd \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/models_output:/app/models_output \
        -v $(pwd)/cache:/app/.cache"
    
    # Add environment file if it exists
    if [ -f .env ]; then
        docker_cmd="$docker_cmd -v $(pwd)/.env:/app/.env:ro"
    else
        print_warning ".env file not found. Creating from template..."
        if [ -f env.example ]; then
            cp env.example .env
            print_info "Created .env from env.example. Please edit it with your API keys."
        fi
    fi
    
    # Add container name and image
    docker_cmd="$docker_cmd --name scotus-ai-temp scotus-ai:latest"
    
    # Add the command
    if [ -n "$command" ]; then
        docker_cmd="$docker_cmd $command $args"
    fi
    
    print_info "Running: $docker_cmd"
    eval $docker_cmd
}

# Function to run with docker-compose
run_compose() {
    local command="$1"
    shift
    local args="$@"
    
    check_docker
    create_directories
    
    if [ -n "$command" ]; then
        print_info "Running with docker-compose: $command $args"
        docker-compose run --rm scotus-ai $command $args
    else
        print_info "Starting docker-compose services..."
        docker-compose up -d
        print_success "Services started. Use 'docker-compose exec scotus-ai bash' to access the container."
    fi
}

# Function to show usage information
show_usage() {
    echo "SCOTUS AI Docker Management Script"
    echo "=================================="
    echo ""
    echo "Usage: $0 [OPTIONS] COMMAND [ARGS...]"
    echo ""
    echo "Commands:"
    echo "  build                     Build the Docker image"
    echo "  shell                     Open an interactive shell in the container"
    echo "  data-pipeline            Run the complete data pipeline"
    echo "  data-pipeline-step STEP  Run a specific pipeline step"
    echo "  encoding                 Run the encoding pipeline"
    echo "  train                    Train the model"
    echo "  hyperparameter-tuning    Run hyperparameter optimization"
    echo "  tune                     Alias for hyperparameter-tuning"
    echo "  check                    Check data status"
    echo "  compose [COMMAND]        Use docker-compose (optional command)"
    echo "  compose-db               Start with database services"
    echo "  clean                    Clean up containers and images"
    echo "  logs                     Show container logs"
    echo ""
    echo "Examples:"
    echo "  $0 build                                    # Build the image"
    echo "  $0 shell                                    # Interactive shell"
    echo "  $0 data-pipeline                           # Run full pipeline"
    echo "  $0 data-pipeline-step scrape-justices      # Run specific step"
    echo "  $0 encoding                                 # Run encoding pipeline"
    echo "  $0 train                                    # Train the model"
    echo "  $0 tune --experiment-name arch_test         # Run hyperparameter tuning"
    echo "  $0 hyperparameter-tuning --n-trials 50     # Run 50 optimization trials"
    echo "  $0 check                                    # Check status"
    echo "  $0 compose                                  # Use docker-compose"
    echo "  $0 compose hyperparameter-tuning --help    # Run tuning with compose"
    echo ""
    echo "Options:"
    echo "  -h, --help               Show this help message"
    echo ""
}

# Main script logic
case "${1:-}" in
    build)
        check_docker
        build_image
        ;;
    shell|bash)
        run_container "bash"
        ;;
    data-pipeline)
        shift
        run_container "data-pipeline" "$@"
        ;;
    data-pipeline-step)
        if [ -z "$2" ]; then
            print_error "Please specify a step name"
            echo "Available steps: scrape-justices, scrape-bios, download-scdb, process-cases, scrape-cases, process-bios, case-metadata, case-descriptions, dataset"
            exit 1
        fi
        shift
        run_container "data-pipeline-step" "$@"
        ;;
    encoding)
        shift
        run_container "encoding" "$@"
        ;;
    encode-bios)
        shift
        run_container "encode-bios" "$@"
        ;;
    encode-descriptions)
        shift
        run_container "encode-descriptions" "$@"
        ;;
    train)
        shift
        run_container "train" "$@"
        ;;
    hyperparameter-tuning|tune)
        if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
            print_info "Hyperparameter Tuning Options:"
            echo "  --experiment-name NAME    Name for the experiment (required)"
            echo "  --n-trials N             Number of trials (default: from config)"
            echo "  --dataset-file PATH      Path to dataset file"
            echo "  --timeout SECONDS        Timeout in seconds"
            echo "  --storage URL            Storage backend (e.g., sqlite:///study.db)"
            echo ""
            echo "Examples:"
            echo "  $0 tune --experiment-name arch_test --n-trials 50"
            echo "  $0 hyperparameter-tuning --experiment-name lr_study --n-trials 100"
            echo "  $0 tune --experiment-name test --timeout 3600"
            echo ""
            exit 0
        fi
        shift
        run_container "hyperparameter-tuning" "$@"
        ;;
    check)
        shift
        run_container "check" "$@"
        ;;
    compose)
        shift
        run_compose "$@"
        ;;
    compose-db)
        check_docker
        create_directories
        print_info "Starting services with database..."
        docker-compose --profile database up -d
        print_success "Services with database started"
        ;;
    clean)
        print_info "Cleaning up Docker containers and images..."
        docker-compose down -v 2>/dev/null || true
        docker container rm scotus-ai-temp 2>/dev/null || true
        docker image rm scotus-ai:latest 2>/dev/null || true
        print_success "Cleanup completed"
        ;;
    logs)
        if docker-compose ps scotus-ai >/dev/null 2>&1; then
            docker-compose logs -f scotus-ai
        else
            print_error "No running scotus-ai container found via docker-compose"
        fi
        ;;
    -h|--help|help)
        show_usage
        ;;
    "")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac 
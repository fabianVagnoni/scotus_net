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
    # Check if NVIDIA drivers are available on the host
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        print_warning "NVIDIA drivers not found. GPU features will be disabled."
        print_info "Install NVIDIA drivers first if you have a GPU."
        return 1
    fi
    
    # Check if nvidia-container-runtime is available
    if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
        print_warning "nvidia-container-runtime not found."
        print_info "Install nvidia-container-runtime for GPU support."
        return 1
    fi
    
    # Test if Docker can use GPUs with the --gpus flag
    if docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        print_info "Docker GPU support detected"
        return 0
    elif docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        print_info "Docker GPU support detected"
        return 0
    else
        print_warning "NVIDIA Docker support not available. GPU features will be disabled."
        print_info "To enable GPU support, configure Docker daemon:"
        print_info "1. Create/edit /etc/docker/daemon.json:"
        print_info '   {"default-runtime": "nvidia", "runtimes": {"nvidia": {"path": "/usr/bin/nvidia-container-runtime", "runtimeArgs": []}}}'
        print_info "2. Restart Docker: sudo systemctl restart docker"
        print_info "3. Test with: docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi"
        return 1
    fi
}

# Function to setup GPU support (helper function)
setup_gpu_support() {
    print_info "Setting up GPU support..."
    print_info "This will configure Docker to use NVIDIA runtime."
    
    # Check if running as root or with sudo
    if [ "$EUID" -ne 0 ]; then
        print_error "GPU setup requires root privileges. Please run with sudo."
        return 1
    fi
    
    # Backup existing daemon.json if it exists
    if [ -f /etc/docker/daemon.json ]; then
        print_info "Backing up existing daemon.json..."
        cp /etc/docker/daemon.json /etc/docker/daemon.json.backup
    fi
    
    # Create daemon.json with NVIDIA runtime
    print_info "Creating /etc/docker/daemon.json..."
    cat > /etc/docker/daemon.json << EOF
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
    
    print_info "Restarting Docker daemon..."
    systemctl restart docker
    
    print_info "Testing GPU support..."
    if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >/dev/null 2>&1; then
        print_success "GPU support configured successfully!"
        return 0
    else
        print_error "GPU support test failed. Please check the setup."
        return 1
    fi
}

# Function to create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    mkdir -p data/raw data/processed data/augmented logs logs/hyperparameter_tunning_logs logs/training_logs models_output/contrastive_justice cache
    
    # Ensure logs and models_output directories are writable by making them world-writable
    # This is necessary for the container's non-root user to write log files and save models
    chmod 777 logs logs/hyperparameter_tunning_logs logs/training_logs models_output 2>/dev/null || {
        print_warning "Could not set write permissions on logs/models directories"
        print_warning "You may need to run: sudo chmod 777 logs/ logs/hyperparameter_tunning_logs/ logs/training_logs/ models_output/"
    }
    
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
    
    # Add GPU support if available - check for nvidia runtime using JSON format
    if docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"' 2>/dev/null; then
        print_info "NVIDIA runtime detected - enabling GPU support"
        docker_cmd="$docker_cmd --gpus all"
    elif check_nvidia_docker; then
        docker_cmd="$docker_cmd --gpus all"
    else
        print_warning "GPU support not available - running in CPU mode"
    fi
    
    # Add volume mounts
    docker_cmd="$docker_cmd \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/models_output:/app/models \
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
    echo "  train                    Train the model with optimized hyperparameters"
echo "  hyperparameter-tuning    Run hyperparameter optimization"
echo "  tune                     Alias for hyperparameter-tuning"
    echo "  pretrain                 Run contrastive justice pretraining"
    echo "  pretrain-tune            Run pretraining hyperparameter optimization"
    echo "  augmentation            Run the augmentation pipeline"
    echo "  check                    Check data status"
    echo "  setup-gpu                Configure Docker for GPU support (requires sudo)"
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
    echo "  $0 train --experiment-name production_v1    # Train the model"
    echo "  $0 tune --experiment-name arch_test         # Run hyperparameter tuning"
    echo "  $0 hyperparameter-tuning --n-trials 50     # Run 50 optimization trials"
    echo "  $0 pretrain                                 # Run contrastive pretraining"
    echo "  $0 pretrain-tune --experiment-name pre_test # Run pretraining optimization"
    echo "  $0 augmentation                             # Run full augmentation pipeline"
    echo "  $0 augmentation --bios-only                 # Only augment justice bios"
    echo "  $0 augmentation --descriptions-only         # Only augment case descriptions"
    echo "  sudo $0 setup-gpu                          # Configure GPU support"
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
    augmentation)
        shift
        run_container "augmentation" "$@"
        ;;
    train)
        if [ "$2" = "--help" ] || [ "$2" = "-h" ]; then
            print_info "Training Options:"
            echo "  --experiment-name NAME    Name for the experiment (required)"
            echo ""
            echo "Examples:"
            echo "  $0 train --experiment-name production_v1"
            echo "  $0 train --experiment-name optimal_trial_31"
            echo ""
            echo "The training will:"
            echo "  - Use optimized hyperparameters from config.env"
            echo "  - Save logs to logs/training_logs/"
            echo "  - Save best model based on combined metric"
            echo "  - Follow three-step fine-tuning strategy"
            echo ""
            exit 0
        fi
        shift
        run_container "train" "$@"
        ;;
    hyperparameter-tuning|tune)
        if [ "$2" = "--help" ] || [ "$2" = "-h" ]; then
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
    pretrain)
        if [ "$2" = "--help" ] || [ "$2" = "-h" ]; then
            print_info "Contrastive Justice Pretraining:"
            echo "  Runs contrastive learning on justice biographies"
            echo "  Uses configuration from scripts/pretraining/config.env"
            echo ""
            echo "Examples:"
            echo "  $0 pretrain"
            echo ""
            exit 0
        fi
        shift
        run_container "pretrain" "$@"
        ;;
    pretrain-tune)
        if [ "$2" = "--help" ] || [ "$2" = "-h" ]; then
            print_info "Pretraining Hyperparameter Tuning Options:"
            echo "  --experiment-name NAME    Name for the experiment (required)"
            echo "  --n-trials N             Number of trials (default: from config)"
            echo "  --config-file PATH       Path to configuration file"
            echo ""
            echo "Examples:"
            echo "  $0 pretrain-tune --experiment-name pre_test --n-trials 50"
            echo "  $0 pretrain-tune --experiment-name lr_study"
            echo ""
            exit 0
        fi
        shift
        run_container "pretrain-tune" "$@"
        ;;
    check)
        shift
        run_container "check" "$@"
        ;;
    setup-gpu)
        setup_gpu_support
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
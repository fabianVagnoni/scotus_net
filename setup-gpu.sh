#!/bin/bash

# SCOTUS AI GPU Setup Script
# ===========================
# This script configures Docker to use NVIDIA GPU support

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

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        print_error "NVIDIA drivers not found. Please install NVIDIA drivers first."
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Check if nvidia-container-runtime is installed
    if ! command -v nvidia-container-runtime >/dev/null 2>&1; then
        print_warning "nvidia-container-runtime not found."
        print_info "Installing nvidia-container-runtime..."
        
        # Add NVIDIA repository
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        apt-get update
        apt-get install -y nvidia-container-toolkit
        
        print_success "nvidia-container-toolkit installed"
    fi
    
    print_success "Prerequisites check passed"
}

# Configure Docker daemon
configure_docker() {
    print_info "Configuring Docker daemon for GPU support..."
    
    # Backup existing daemon.json if it exists
    if [ -f /etc/docker/daemon.json ]; then
        print_info "Backing up existing daemon.json..."
        cp /etc/docker/daemon.json /etc/docker/daemon.json.backup.$(date +%Y%m%d_%H%M%S)
        print_info "Backup saved as daemon.json.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Create or update daemon.json
    print_info "Creating/updating /etc/docker/daemon.json..."
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
    
    print_success "Docker daemon configuration updated"
}

# Restart Docker service
restart_docker() {
    print_info "Restarting Docker daemon..."
    systemctl restart docker
    
    # Wait for Docker to be ready
    sleep 3
    
    # Check if Docker is running
    if systemctl is-active --quiet docker; then
        print_success "Docker restarted successfully"
    else
        print_error "Docker failed to restart. Please check the logs."
        exit 1
    fi
}

# Test GPU support
test_gpu_support() {
    print_info "Testing GPU support..."
    
    # Pull NVIDIA CUDA image if not present - using valid tag
    if ! docker image inspect nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 >/dev/null 2>&1; then
        print_info "Pulling NVIDIA CUDA test image..."
        docker pull nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
    fi
    
    # Test GPU access
    if docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        print_success "GPU support test passed!"
        print_info "Running nvidia-smi in container to show GPU info:"
        docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi
    else
        print_error "GPU support test failed. Please check the configuration."
        exit 1
    fi
}

# Main function
main() {
    echo "SCOTUS AI GPU Setup Script"
    echo "=========================="
    echo ""
    
    check_root
    check_prerequisites
    configure_docker
    restart_docker
    test_gpu_support
    
    echo ""
    print_success "GPU support setup completed successfully!"
    print_info "You can now use GPU acceleration with:"
    print_info "  ./docker-run.sh tune --experiment-name test"
    print_info "  docker run --rm --gpus all scotus-ai:latest"
    echo ""
}

# Run main function
main "$@" 
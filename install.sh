#!/bin/bash
# Label Studio + YOLO Auto-Trainer - One-Line Installer
# Usage: curl -fsSL https://.../install.sh | bash

set -e

REPO_URL="https://github.com/n00n0i/labelstudio-yolo-autotrainer"
INSTALL_DIR="${INSTALL_DIR:-$HOME/labelstudio-yolo}"
VERSION="${VERSION:-main}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Progress bar function
progress_bar() {
    local duration=$1
    local prefix=$2
    local width=50
    local increment=$((100 / width))
    
    printf "${prefix} ["
    for i in $(seq 1 $width); do
        sleep $(echo "scale=3; $duration / $width" | bc)
        if [ $i -eq $width ]; then
            printf "="
        else
            printf "="
        fi
    done
    printf "] 100%%\n"
}

# Spinner function
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Logging functions
log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_step() {
    echo -e "${BLUE}→${NC} ${BOLD}$1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} ${BOLD}$1${NC}"
}

# Print banner
print_banner() {
    clear
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🏷️  Label Studio + 🤖 YOLO Auto-Trainer                  ║
║                                                              ║
║     Automatic model training when annotations reach          ║
║     the threshold!                                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

EOF
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            echo "$NAME"
        else
            echo "Linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS"
    else
        echo "Unknown"
    fi
}

# Check prerequisites with visual feedback
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    local checks=("docker" "docker-compose" "git")
    local missing=()
    
    for cmd in "${checks[@]}"; do
        printf "  Checking %-20s" "$cmd..."
        if command_exists "$cmd"; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
            missing+=("$cmd")
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        echo ""
        log_error "Missing required tools: ${missing[*]}"
        echo ""
        echo "Please install:"
        echo "  • Docker: https://docs.docker.com/get-docker/"
        echo "  • Docker Compose: https://docs.docker.com/compose/install/"
        echo "  • Git: https://git-scm.com/downloads"
        exit 1
    fi
    
    # Check Docker is running
    printf "  Checking Docker daemon..."
    if docker info >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "All prerequisites met!"
}

# Check GPU with visual feedback
check_gpu() {
    log_step "Checking GPU support..."
    
    printf "  Detecting NVIDIA GPU..."
    if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        GPU_AVAILABLE=true
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo ""
        echo -e "${CYAN}  GPU(s) detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
            echo -e "    ${GREEN}•${NC} $line"
        done
        echo ""
    else
        echo -e "${YELLOW}⚠${NC}"
        GPU_AVAILABLE=false
        log_warn "No NVIDIA GPU detected. Training will use CPU (slower)."
        echo ""
        echo "To enable GPU:"
        echo "  1. Install NVIDIA drivers"
        echo "  2. Install NVIDIA Container Toolkit"
        echo "     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo ""
    fi
}

# Download with progress
download_repo() {
    log_step "Downloading Label Studio YOLO Auto-Trainer..."
    
    if [ -d "$INSTALL_DIR" ]; then
        log_warn "Directory $INSTALL_DIR already exists"
        read -p "  Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Installation cancelled"
            exit 0
        fi
        rm -rf "$INSTALL_DIR"
    fi
    
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    printf "  Cloning repository..."
    git clone --depth 1 "$REPO_URL" . >/dev/null 2>&1 &
    spinner $!
    echo -e "${GREEN}✓${NC}"
    
    log_success "Downloaded to $INSTALL_DIR"
}

# Setup environment with visual feedback
setup_environment() {
    log_step "Setting up environment..."
    
    cd "$INSTALL_DIR"
    
    if [ ! -f .env ]; then
        printf "  Generating secure tokens..."
        
        LABEL_STUDIO_TOKEN=$(openssl rand -hex 32 2>/dev/null || cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 64 | head -n 1)
        
        cat > .env << EOF
# Label Studio YOLO Auto-Trainer
# Generated: $(date)

# Label Studio Configuration
LABEL_STUDIO_TOKEN=$LABEL_STUDIO_TOKEN
LABEL_STUDIO_USERNAME=admin@example.com
LABEL_STUDIO_PASSWORD=admin

# Training Configuration
TRAINING_THRESHOLD=100

# Directories
DATASET_DIR=./datasets
MODELS_DIR=./models

# GPU Settings
GPU_ENABLED=${GPU_AVAILABLE:-false}
CUDA_VISIBLE_DEVICES=0
EOF
        
        echo -e "${GREEN}✓${NC}"
        log_success "Environment file created"
        
        echo ""
        echo -e "${CYAN}  Generated credentials:${NC}"
        echo -e "    Label Studio Token: ${YELLOW}${LABEL_STUDIO_TOKEN:0:20}...${NC}"
        echo ""
        log_warn "Save these credentials securely!"
    else
        log_info "Environment file already exists"
    fi
}

# Create directories
setup_directories() {
    log_step "Creating directories..."
    
    mkdir -p "$INSTALL_DIR"/{datasets,models,labelstudio-data,logs}
    
    log_success "Directories created"
}

# Pull images with progress
pull_images() {
    log_step "Pulling Docker images..."
    
    cd "$INSTALL_DIR"
    
    echo "  This may take a few minutes..."
    docker-compose pull 2>&1 | while read line; do
        if [[ $line == *"Pulling"* ]]; then
            printf "  %s\n" "$line"
        fi
    done
    
    log_success "Images pulled"
}

# Build images
build_images() {
    log_step "Building custom images..."
    
    cd "$INSTALL_DIR"
    
    docker-compose build --no-rm 2>&1 | while read line; do
        if [[ $line == *"Step"* ]] || [[ $line == *"Successfully"* ]]; then
            printf "  %s\n" "$line"
        fi
    done
    
    log_success "Images built"
}

# Start services
start_services() {
    log_step "Starting services..."
    
    cd "$INSTALL_DIR"
    
    docker-compose up -d 2>&1 | while read line; do
        if [[ $line == *"Creating"* ]] || [[ $line == *"Started"* ]]; then
            printf "  %s\n" "$line"
        fi
    done
    
    log_success "Services started"
}

# Wait for services with progress
wait_for_services() {
    log_step "Waiting for services to be ready..."
    
    local services=("labelstudio:8080" "auto-trainer:8000")
    local max_attempts=30
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        printf "  Waiting for %-20s" "$name..."
        
        local attempt=1
        while [ $attempt -le $max_attempts ]; do
            if curl -fsSL "http://localhost:$port/health" >/dev/null 2>&1 || \
               curl -fsSL "http://localhost:$port" >/dev/null 2>&1; then
                echo -e "${GREEN}✓${NC}"
                break
            fi
            sleep 2
            attempt=$((attempt + 1))
        done
        
        if [ $attempt -gt $max_attempts ]; then
            echo -e "${YELLOW}⚠${NC}"
            log_warn "$name may not be fully ready yet"
        fi
    done
}

# Print completion with style
print_completion() {
    local width=70
    
    echo ""
    printf "╔%${width}s╗\n" "" | tr " " "="
    printf "║%s║\n" "$(printf '%*s' $(( (width - 20) / 2 )) '')🎉 Installation Complete!"
    printf "╚%${width}s╝\n" "" | tr " " "="
    echo ""
    
    echo -e "${BOLD}📁 Installation Directory:${NC}"
    echo "   $INSTALL_DIR"
    echo ""
    
    echo -e "${BOLD}🌐 Access URLs:${NC}"
    printf "  ${CYAN}%-20s${NC} %s\n" "Label Studio:" "http://localhost:8080"
    printf "  ${CYAN}%-20s${NC} %s\n" "Auto-Trainer API:" "http://localhost:8000"
    printf "  ${CYAN}%-20s${NC} %s\n" "Training Status:" "http://localhost:8000/training/status"
    echo ""
    
    echo -e "${BOLD}🔑 Default Credentials:${NC}"
    printf "  ${CYAN}%-20s${NC} %s\n" "Label Studio:" "admin@example.com / admin"
    echo ""
    
    echo -e "${BOLD}🚀 Quick Start:${NC}"
    echo "   1. Open http://localhost:8080"
    echo "   2. Login with credentials above"
    echo "   3. Create a project with bounding box labels"
    echo "   4. Start annotating!"
    echo "   5. Training auto-starts at 100 annotations"
    echo ""
    
    echo -e "${BOLD}🛠️  Useful Commands:${NC}"
    printf "  ${YELLOW}%-35s${NC} %s\n" "cd $INSTALL_DIR" "# Go to install directory"
    printf "  ${YELLOW}%-35s${NC} %s\n" "docker-compose logs -f" "# View logs"
    printf "  ${YELLOW}%-35s${NC} %s\n" "docker-compose down" "# Stop services"
    printf "  ${YELLOW}%-35s${NC} %s\n" "docker-compose up -d" "# Start services"
    echo ""
    
    printf "╔%${width}s╗\n" "" | tr " " "="
    printf "║%s║\n" "$(printf '%*s' $(( (width - 35) / 2 )) '')⚠️  FOR RESEARCH USE ONLY"
    printf "╚%${width}s╝\n" "" | tr " " "="
    echo ""
}

# Main installation
main() {
    print_banner
    
    echo -e "${BOLD}Installing Label Studio + YOLO Auto-Trainer${NC}"
    echo -e "Version: ${CYAN}$VERSION${NC}"
    echo -e "OS: ${CYAN}$(detect_os)${NC}"
    echo -e "Install Directory: ${CYAN}$INSTALL_DIR${NC}"
    echo ""
    
    check_prerequisites
    check_gpu
    download_repo
    setup_environment
    setup_directories
    pull_images
    build_images
    start_services
    wait_for_services
    
    print_completion
}

# Handle interruption
trap 'echo "" ; log_error "Installation interrupted" ; exit 1' INT TERM

# Run
main

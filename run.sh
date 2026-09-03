#!/usr/bin/env bash
# ==============================================================================
# BookMind — Project Runner Script
# ==============================================================================

set -e

# Color definitions
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Determine project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

print_header() {
    echo -e "${CYAN}${BOLD}"
    echo "======================================================================"
    echo "                 BookMind Project Launcher                           "
    echo "======================================================================"
    echo -e "${NC}"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    print_header
    echo -e "${BOLD}Usage:${NC} ./run.sh [MODE] [OPTIONS]"
    echo ""
    echo -e "${BOLD}Available Modes:${NC}"
    echo -e "  ${GREEN}connected${NC}   (Default) Run local Flask app + starts Qdrant & Redis via Docker"
    echo -e "  ${GREEN}docker${NC}      Run entire stack via Docker Compose (app + qdrant + redis)"
    echo -e "  ${GREEN}test${NC}        Run test suite with pytest"
    echo -e "  ${GREEN}ingest${NC}      Run ingestion script (pass ingestion arguments)"
    echo -e "  ${GREEN}help${NC}        Display this help message"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  ./run.sh                  # Run connected mode (default)"
    echo "  ./run.sh connected        # Run connected mode with Qdrant + Redis"
    echo "  ./run.sh docker           # Run full docker stack"
    echo "  ./run.sh test             # Run pytest suite"
    echo '  ./run.sh ingest data/book.pdf --book-id sample --title "Sample Book" --author "Author"'
    echo ""
}

ensure_env() {
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_info "Creating .env from .env.example..."
            cp .env.example .env
            print_success ".env created successfully."
        else
            print_error ".env.example file not found!"
            exit 1
        fi
    fi
}

setup_venv() {
    if [ ! -d ".venv" ]; then
        print_info "Creating Python virtual environment in .venv..."
        python3 -m venv .venv
        print_success "Virtual environment created."
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Check if key dependencies are installed
    if ! python -c "import flask, langgraph, pydantic" &>/dev/null; then
        print_info "Installing dependencies from requirements.txt..."
        pip install --upgrade pip -q
        pip install -r requirements.txt
        print_success "Dependencies installed."
    fi
}



run_connected() {
    print_header
    ensure_env
    setup_venv

    # Check if docker is available
    if ! command -v docker &>/dev/null; then
        print_error "Docker is required for connected mode (Qdrant & Redis), but docker command was not found."
        exit 1
    fi

    print_info "Starting backend services (Qdrant & Redis) via Docker Compose..."
    docker compose up -d qdrant redis || docker-compose up -d qdrant redis

    print_info "Starting BookMind in ${BOLD}CONNECTED MODE${NC}..."
    echo -e "${GREEN}${BOLD}Server running at: http://localhost:5000${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop.${NC}\n"

    python -m app.main
}

run_docker() {
    print_header
    ensure_env

    if ! command -v docker &>/dev/null; then
        print_error "Docker is required for docker mode, but docker command was not found."
        exit 1
    fi

    print_info "Building and starting full BookMind stack via Docker Compose..."
    echo -e "${GREEN}${BOLD}Server will be running at: http://localhost:5000${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop containers.${NC}\n"

    docker compose up --build || docker-compose up --build
}

run_tests() {
    print_header
    ensure_env
    setup_venv

    if ! python -c "import pytest" &>/dev/null; then
        print_info "Installing development dependencies..."
        pip install -r requirements-dev.txt
    fi

    print_info "Running pytest suite..."
    pytest "$@"
}

run_ingest() {
    ensure_env
    setup_venv

    print_info "Running ingestion script..."
    python -m scripts.ingestion.ingest "$@"
}

# Main routing logic
MODE="${1:-connected}"

case "$MODE" in
    connected)
        run_connected
        ;;
    docker)
        run_docker
        ;;
    test|tests)
        shift
        run_tests "$@"
        ;;
    ingest)
        shift
        run_ingest "$@"
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        print_error "Unknown mode: $MODE"
        echo ""
        show_help
        exit 1
        ;;
esac

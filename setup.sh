#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# setup.sh — One-command setup for the Playlist-to-Book pipeline
#
# What this does:
#   1. Creates a Python virtual environment
#   2. Installs Python dependencies
#   3. Installs Ollama (if not already installed)
#   4. Pulls the default local model (mistral)
#   5. Copies .env.example → .env (if .env doesn't exist)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ─────────────────────────────────────────────────────────────────────

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }

echo ""
echo "════════════════════════════════════════════════"
echo "  Playlist → Book Pipeline — Setup"
echo "════════════════════════════════════════════════"
echo ""

# ── 1. Python virtual environment ────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating Python virtual environment..."
    python3 -m venv .venv
else
    info "Virtual environment already exists."
fi

source .venv/bin/activate
info "Activated virtual environment."

# ── 2. Python dependencies ───────────────────────────────────────────
info "Installing Python dependencies..."
pip install -r requirements.txt --quiet
info "Python dependencies installed."

# ── 3. Ollama ────────────────────────────────────────────────────────
if command -v ollama &> /dev/null; then
    info "Ollama is already installed ($(ollama --version 2>/dev/null || echo 'version unknown'))."
else
    warn "Ollama is not installed. Installing now..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  On macOS, please install Ollama from: https://ollama.com/download"
        echo "  Then re-run this script."
        exit 1
    else
        echo "  Unsupported OS. Please install Ollama manually: https://ollama.com/download"
        exit 1
    fi

    if command -v ollama &> /dev/null; then
        info "Ollama installed successfully."
    else
        error "Ollama installation failed. Install manually: https://ollama.com/download"
        exit 1
    fi
fi

# ── 4. Start Ollama if not running ───────────────────────────────────
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    info "Ollama server is already running."
else
    warn "Starting Ollama server in the background..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        info "Ollama server started."
    else
        warn "Could not start Ollama server. You may need to run 'ollama serve' manually."
    fi
fi

# ── 5. Pull default model ───────────────────────────────────────────
MODEL="mistral"
if ollama list 2>/dev/null | grep -q "$MODEL"; then
    info "Model '$MODEL' is already pulled."
else
    info "Pulling model '$MODEL' (this may take a few minutes on first run)..."
    ollama pull "$MODEL"
    info "Model '$MODEL' pulled successfully."
fi

# ── 6. Environment file ─────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    info "Created .env from .env.example."
    warn "Edit .env to add your API keys (optional — Ollama works without them)."
else
    info ".env file already exists."
fi

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo -e "  ${GREEN}Setup complete!${NC}"
echo "════════════════════════════════════════════════"
echo ""
echo "  Quick start:"
echo "    source .venv/bin/activate"
echo "    python main.py --url \"https://youtube.com/watch?v=...&list=...\" --dry-run"
echo ""
echo "  The pipeline is configured to use Ollama (local, free) by default."
echo "  To use cloud providers, add API keys to .env and change"
echo "  primary_provider in config/default.yaml."
echo ""

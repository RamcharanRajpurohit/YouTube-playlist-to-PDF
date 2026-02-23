#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# setup.sh — One-command setup for the Playlist-to-Book pipeline
#
# What this does:
#   1. Creates a Python virtual environment
#   2. Installs Python dependencies
#   3. Copies .env.example → .env (if .env doesn't exist)
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

# ── 3. Environment file ─────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    info "Created .env from .env.example."
    warn "Edit .env to add your API keys (e.g. GEMINI_API_KEY)."
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
echo "  The pipeline is configured to use Gemini by default."
echo "  Make sure to add your GEMINI_API_KEY to .env."
echo ""

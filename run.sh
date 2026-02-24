#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run.sh — Run the Playlist-to-Book pipeline via Docker
#
# Usage:
#   ./run.sh                     # interactive mode (prompts for API key)
#   ./run.sh --dry-run           # only generate TOC, skip chapter writing
# ─────────────────────────────────────────────────────────────────────

set -e

IMAGE="playlist-to-book"

echo "══════════════════════════════════════════════════"
echo "  Playlist → Book Pipeline"
echo "══════════════════════════════════════════════════"
echo ""

# Build (or rebuild) the image — Docker layer caching makes this fast if nothing changed
echo "[1/3] Building Docker image..."
echo ""
docker build -t "$IMAGE" .
echo ""
echo "[1/3] Docker image ready."

echo "[2/3] Preparing output directory..."
mkdir -p "$(pwd)/output"

# Pass through API key from host .env or environment if available
ENV_ARGS=""
if [ -n "$GEMINI_API_KEY" ]; then
    ENV_ARGS="-e GEMINI_API_KEY=$GEMINI_API_KEY"
    echo "       GEMINI_API_KEY found in environment."
elif [ -f ".env" ]; then
    ENV_ARGS="--env-file .env"
    echo "       Loading API keys from .env file."
else
    echo "       No API key found — you will be prompted inside the container."
fi

echo "[3/3] Starting pipeline..."
echo ""

# :z flag is needed on SELinux systems (Fedora, RHEL) to allow container writes
docker run --rm -it --user "$(id -u):$(id -g)" $ENV_ARGS -v "$(pwd)/output:/app/output:z" "$IMAGE" "$@"

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run.sh — Run the Playlist-to-Book pipeline via Docker
#
# Usage:
#   ./run.sh                     # interactive mode (prompts for everything)
#   ./run.sh --url "https://..."  # pass arguments directly
# ─────────────────────────────────────────────────────────────────────

set -e

IMAGE="playlist-to-book"

# Build (or rebuild) the image — Docker layer caching makes this fast if nothing changed
docker build -q -t "$IMAGE" . > /dev/null

mkdir -p "$(pwd)/output"

# Pass through API key from host .env or environment if available
ENV_ARGS=""
if [ -n "$GEMINI_API_KEY" ]; then
    ENV_ARGS="-e GEMINI_API_KEY=$GEMINI_API_KEY"
elif [ -f ".env" ]; then
    ENV_ARGS="--env-file .env"
fi

# :z flag is needed on SELinux systems (Fedora, RHEL) to allow container writes
docker run --rm -it --user "$(id -u):$(id -g)" $ENV_ARGS -v "$(pwd)/output:/app/output:z" "$IMAGE" "$@"

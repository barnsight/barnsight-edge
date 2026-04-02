#!/bin/bash

# Build the BarnSight Edge Docker image
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

IMAGE_NAME="${BARN_SIGHT_IMAGE:-barnsight-edge}"

log_info "Building Docker image: ${IMAGE_NAME}"
log_info "This may take a few minutes on first build..."

docker build -t "${IMAGE_NAME}:latest" .

log_info "Build complete: ${IMAGE_NAME}:latest"
log_info "Start cameras with: ./scripts/docker-up.sh"

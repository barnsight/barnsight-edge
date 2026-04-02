#!/bin/bash

# Start all camera containers (background mode)
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

REBUILD=""
CAMERA=""

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "  -r, --rebuild    Rebuild images before starting"
  echo "  -c, --camera     Start a single camera (e.g. cam-barn-01)"
  echo "  -h, --help       Show this help"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--rebuild) REBUILD="--build"; shift ;;
    -c|--camera)  CAMERA="$2"; shift 2 ;;
    -h|--help)    usage ;;
    *)            echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -n "$CAMERA" ]; then
  log_info "Starting camera: $CAMERA"
  docker compose up -d $REBUILD "$CAMERA"
else
  log_info "Starting all cameras..."
  docker compose up -d $REBUILD
fi

log_info "Running containers:"
docker compose ps

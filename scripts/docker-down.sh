#!/bin/bash

# Stop all or specific camera containers
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

CAMERA=""

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "  -c, --camera NAME   Stop a single camera (e.g. cam-barn-01)"
  echo "  -a, --all           Stop and remove all containers (default)"
  echo "  -h, --help          Show this help"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--camera) CAMERA="$2"; shift 2 ;;
    -a|--all)    shift ;;
    -h|--help)   usage ;;
    *)           echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -n "$CAMERA" ]; then
  log_info "Stopping camera: $CAMERA"
  docker compose stop "$CAMERA"
else
  log_info "Stopping all cameras..."
  docker compose down
fi

log_info "Done."

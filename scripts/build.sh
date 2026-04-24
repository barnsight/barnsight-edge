#!/usr/bin/env bash

# Prepare the local BarnSight Edge development/runtime environment.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1" >&2
}

if ! command -v uv >/dev/null 2>&1; then
  log_error "uv is required. Install it from https://github.com/astral-sh/uv"
  exit 1
fi

log_info "Syncing dependencies..."
uv sync

log_info "Running tests..."
uv run pytest tests/ -v

log_info "Checking Python compilation..."
uv run python -m compileall src

log_info "Build completed successfully."

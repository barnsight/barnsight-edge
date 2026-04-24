#!/usr/bin/env bash

# Remove local generated files without touching source or credentials.
set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_info "Removing Python caches..."
find . -type d \
  \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".mypy_cache" \) \
  -prune -exec rm -rf {} +

log_info "Removing coverage artifacts..."
rm -rf .coverage .coverage.* htmlcov coverage.xml

log_info "Clean completed."

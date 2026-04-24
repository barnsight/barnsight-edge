#!/usr/bin/env bash

# Run the BarnSight Edge worker locally.
set -euo pipefail

uv run python -m src.main

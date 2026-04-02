#!/bin/bash

# Follow logs for one or all cameras
set -euo pipefail

CAMERA=""
LINES=50

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "  -c, --camera NAME   Show logs for one camera (e.g. cam-barn-01)"
  echo "  -n, --lines NUM     Number of lines to show (default: 50)"
  echo "  -h, --help          Show this help"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--camera) CAMERA="$2"; shift 2 ;;
    -n|--lines)  LINES="$2"; shift 2 ;;
    -h|--help)   usage ;;
    *)           echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ -n "$CAMERA" ]; then
  docker compose logs -f --tail="$LINES" "$CAMERA"
else
  docker compose logs -f --tail="$LINES"
fi

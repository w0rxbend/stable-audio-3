#!/usr/bin/env bash
# Run from any directory; flags after the script override these batch defaults.
set -euo pipefail

SA3_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SA3_REPO_DIR"

exec python3 generate_from_lofi_presets.py prompts/afterhours_144.jsonl \
    --backend tflite --model medium --duration 300 --threads 8 \
    --steps 8 --cfg-scale 1.0 --dit-precision w8a8 --decoder same-s \
    --format mp3 --out-dir output/afterhours_144 "$@"

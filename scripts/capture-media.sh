#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

ZENN_IMAGES_DIR="${ZENN_IMAGES_DIR:-$ROOT/../m-zenn-dev/images}"
mkdir -p .media-build "$ZENN_IMAGES_DIR"

python scripts/render_terminal_png.py \
  .media-build/before_after.txt \
  "$ZENN_IMAGES_DIR/slm-humor-distill-before-after.png" 1024

python scripts/render_terminal_png.py \
  .media-build/train_bench.txt \
  "$ZENN_IMAGES_DIR/slm-humor-distill-train-bench.png" 880

echo "Saved to $ZENN_IMAGES_DIR/slm-humor-distill-*.png"

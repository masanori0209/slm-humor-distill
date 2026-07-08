#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

MODEL="${MODEL:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}"
ADAPTER="${ADAPTER:-adapters/humor-lora}"
OUT="${OUT:-fused_model/humor-1.5b-fused}"

echo "=== Fuse LoRA adapter into base model ==="
mlx_lm.fuse \
  --model "$MODEL" \
  --adapter-path "$ADAPTER" \
  --save-path "$OUT"

echo "Fused model saved to: $OUT"

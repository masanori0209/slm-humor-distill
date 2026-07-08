#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

echo "=== LoRA fine-tuning (distillation via SFT) ==="
mlx_lm.lora --config configs/lora.yaml

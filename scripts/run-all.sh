#!/usr/bin/env bash
# デモ全体の実行順（初回は teacher 14B の DL で時間がかかります）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  uv venv --python 3.12 .venv
  uv pip install --python .venv -r requirements.txt
fi
source .venv/bin/activate

echo "=== 1/5 Teacher data generation ==="
python scripts/generate_teacher_data.py "$@"

echo "=== 2/5 Prepare train/valid split ==="
python scripts/prep_train_data.py

echo "=== 3/5 LoRA training ==="
bash scripts/train_lora.sh

echo "=== 4/5 Fuse adapter ==="
bash scripts/fuse_model.sh

echo "=== 5/5 Before/After eval ==="
python scripts/eval_before_after.py --fused-model fused_model/humor-1.5b-fused
python scripts/bench_mlx.py --model fused_model/humor-1.5b-fused

echo "Done. See reports/ for outputs."

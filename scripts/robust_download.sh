#!/usr/bin/env bash
# HF ダウンロードが数百MBで落ちるので、resume で完了するまで単純再試行する。
# --max-workers 1 で逐次 DL（並行 DL より安定）。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPO="${1:-mlx-community/Qwen2.5-14B-Instruct-4bit}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=30

for attempt in $(seq 1 500); do
  echo "=== attempt $attempt for $REPO at $(date +%T) ==="
  if .venv/bin/hf download "$REPO" --max-workers 1 >> reports/dl_retry.log 2>&1; then
    echo "=== DOWNLOAD COMPLETE at $(date +%T) ==="
    exit 0
  fi
  echo "attempt $attempt ended, resuming in 2s..."
  sleep 2
done
echo "=== GAVE UP ==="
exit 1

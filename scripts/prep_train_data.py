#!/usr/bin/env python3
"""teacher_raw.jsonl を mlx-lm 用 train/valid.jsonl に分割する。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "generated" / "teacher_raw.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "train",
    )
    p.add_argument("--valid-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    random.seed(args.seed)
    random.shuffle(rows)
    n_valid = max(1, int(len(rows) * args.valid_ratio))
    valid = rows[:n_valid]
    train = rows[n_valid:]

    train_rows = [{"messages": r["messages"]} for r in train]
    valid_rows = [{"messages": r["messages"]} for r in valid]

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "valid.jsonl", valid_rows)
    print(f"train={len(train_rows)} valid={len(valid_rows)} -> {args.output_dir}")


if __name__ == "__main__":
    main()

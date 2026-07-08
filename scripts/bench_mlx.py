#!/usr/bin/env python3
"""推論速度・メモリの簡易ベンチ（MLX / Apple Silicon）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import SYSTEM_PROMPT, load_models_config, make_temp_sampler, user_prompt  # noqa: E402
from mlx_lm import generate, load  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="")
    p.add_argument("--adapter", type=str, default="")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "bench_mlx.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_models_config()
    model_path = args.model or cfg["student"]["mlx_repo"]

    if args.adapter:
        model, tokenizer = load(model_path, adapter_path=args.adapter)
        label = f"{model_path} + {args.adapter}"
    else:
        model, tokenizer = load(model_path)
        label = model_path

    topic = "電子レンジが壊れた"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(topic)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    times: list[float] = []
    tokens_approx = 0
    for r in range(args.rounds):
        t0 = time.perf_counter()
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=80,
            sampler=make_temp_sampler(0.8),
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        tokens_approx = len(text)  # ざっくり
        print(f"round {r+1}: {elapsed:.2f}s -> {text[:60]}...")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": label,
        "rounds": args.rounds,
        "avg_sec": sum(times) / len(times),
        "times_sec": times,
        "note": "MLX on Apple Silicon (Metal). Strict CPU-only needs GGUF + llama.cpp -ngl 0.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()

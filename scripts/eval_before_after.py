#!/usr/bin/env python3
"""before（素モデル）/ after（LoRA or fused）を同一お題で比較する。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    SYSTEM_PROMPT,
    load_models_config,
    load_topics,
    make_temp_sampler,
    user_prompt,
)
from mlx_lm import generate, load  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=str, default="", help="LoRA adapter path (after)")
    p.add_argument(
        "--fused-model",
        type=str,
        default="",
        help="Fused model path (after, adapter より優先)",
    )
    p.add_argument("--limit", type=int, default=10)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "before_after.jsonl",
    )
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=80)
    return p.parse_args()


def run_generate(model, tokenizer, topic: str, temp: float, max_tokens: int) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(topic)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    t0 = time.perf_counter()
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_temp_sampler(temp),
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return text.strip(), elapsed


def main() -> None:
    args = parse_args()
    cfg = load_models_config()
    base_repo = cfg["student"]["mlx_repo"]
    topics = load_topics()[: args.limit]

    print(f"Loading base: {base_repo}")
    base_model, tokenizer = load(base_repo)

    after_model = None
    after_label = "none"
    if args.fused_model:
        print(f"Loading fused (after): {args.fused_model}")
        after_model, _ = load(args.fused_model)
        after_label = args.fused_model
    elif args.adapter:
        print(f"Loading base + adapter (after): {args.adapter}")
        after_model, _ = load(base_repo, adapter_path=args.adapter)
        after_label = args.adapter

    rows = []
    for i, topic in enumerate(topics, 1):
        before, t_before = run_generate(
            base_model, tokenizer, topic, args.temperature, args.max_tokens
        )
        after = ""
        t_after = 0.0
        if after_model is not None:
            after, t_after = run_generate(
                after_model, tokenizer, topic, args.temperature, args.max_tokens
            )

        row = {
            "topic": topic,
            "before": before,
            "after": after,
            "time_before_sec": round(t_before, 3),
            "time_after_sec": round(t_after, 3),
            "after_model": after_label,
        }
        rows.append(row)
        print(f"\n[{i}] {topic}")
        print(f"  BEFORE: {before}")
        if after:
            print(f"  AFTER:  {after}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()

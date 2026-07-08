#!/usr/bin/env python3
"""先生モデル（Qwen2.5-14B）で教師データを生成する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    chat_messages,
    is_clean_japanese,
    load_models_config,
    load_topics,
    make_temp_sampler,
    write_jsonl,
)
from mlx_lm import generate, load  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate teacher humor dataset")
    p.add_argument("--limit", type=int, default=0, help="0 = all topics")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "generated" / "teacher_raw.jsonl",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=60)
    p.add_argument(
        "--samples-per-topic",
        type=int,
        default=6,
        help="各お題を何回サンプリングするか（フィルタ後に残った分だけ採用）",
    )
    return p.parse_args()


def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    args = parse_args()
    cfg = load_models_config()
    teacher_repo = cfg["teacher"]["mlx_repo"]
    topics = load_topics()
    if args.limit > 0:
        topics = topics[: args.limit]

    print(f"Loading teacher: {teacher_repo}")
    model, tokenizer = load(teacher_repo)
    sampler = make_temp_sampler(args.temperature)

    rows: list[dict] = []
    kept = 0
    dropped = 0
    for i, topic in enumerate(topics, 1):
        messages = chat_messages(topic)
        prompt = apply_chat_template(tokenizer, messages)
        print(f"[{i}/{len(topics)}] {topic}")
        seen: set[str] = set()
        for _ in range(args.samples_per_topic):
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            text = response.strip()
            if not is_clean_japanese(text) or text in seen:
                dropped += 1
                print(f"  x {text}")
                continue
            seen.add(text)
            rows.append(
                {
                    "topic": topic,
                    "messages": chat_messages(topic, text),
                    "teacher_model": teacher_repo,
                }
            )
            kept += 1
            print(f"  o {text}")
        print()

    write_jsonl(args.output, rows)
    print(f"Wrote {kept} rows (dropped {dropped}) -> {args.output}")


if __name__ == "__main__":
    main()

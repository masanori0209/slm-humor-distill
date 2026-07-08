"""共通ユーティリティ。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"

SYSTEM_PROMPT = """\
あなたは「くすっと笑える一言」を返す日本語のアシスタントです。

ルール:
- 1〜2文、40字以内を目安に
- 励ましすぎない。軽い自虐・皮肉・意外性で笑わせる
- 説教・「頑張りましょう」系は禁止
- 必ず自然な日本語だけで返す（英語・中国語・韓国語を混ぜない）
- ひらがな・カタカナ・漢字・句読点のみを使う
"""


def load_models_config() -> dict[str, Any]:
    with (CONFIGS / "models.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_topics(path: Path | None = None) -> list[str]:
    path = path or DATA / "prompts" / "seed_topics.txt"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def user_prompt(topic: str) -> str:
    return f"状況: {topic}\n一言で、くすっと笑える返事をして。"


def chat_messages(topic: str, assistant: str | None = None) -> list[dict[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(topic)},
    ]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def is_clean_japanese(text: str, max_len: int = 80) -> bool:
    """非日本語（中国語・ハングル・過度なラテン文字）の混入を弾く簡易フィルタ。"""
    import re

    if not text or len(text) > max_len:
        return False
    # ハングル
    if re.search(r"[\uac00-\ud7a3]", text):
        return False
    # キリル文字
    if re.search(r"[\u0400-\u04ff]", text):
        return False
    # ラテン文字（少しは許容するが、多いと英語混入とみなす）
    latin = len(re.findall(r"[A-Za-z]", text))
    if latin > 6:
        return False
    # 日本語らしさ（ひらがな/カタカナが最低限あること）
    kana = len(re.findall(r"[\u3040-\u30ff]", text))
    if kana < 3:
        return False
    # 中国語簡体字に多い記号の混入をざっくり弾く
    if re.search(r"[，。！？；：（）]", text):  # 全角中国語約物
        pass  # 日本語でも全角記号は使うので許容
    return True


def make_temp_sampler(temp: float, top_p: float = 0.9):
    """mlx-lm 0.31 系: generate は temp 引数を受けず sampler を使う。"""
    from mlx_lm.sample_utils import make_sampler

    return make_sampler(temp=temp, top_p=top_p)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# slm-humor-distill-demo

**Qwen2.5-14B を先生に、Qwen2.5-1.5B を「くすっと笑える一言」特化へ蒸留する**最小デモです。

Sakana AI の [TinySwallow-1.5B](https://huggingface.co/SakanaAI/TinySwallow-1.5B)（32B → 1.5B の TAID 蒸留）をオマージュしつつ、家庭の MacBook Air（Apple M5 / 32GB）で **合成データ蒸留（sequence-level distillation）** を再現します。

> **Note:** 本番品質のユーモアモデルではありません。合成お題 20 件規模の教育デモです。厳密な logit 蒸留（TAID）の再現でもありません。

解説記事（Zenn）: （公開後に追記）

## できること

- 先生（14B 4bit）が「くすっと返し」を生成
- 生徒（1.5B 4bit）を MLX LoRA で微調整（= 蒸留）
- before / after を同一お題で比較
- MLX 上での推論ベンチ（tok/s 近似）

## モデル構成

| 役割 | モデル | MLX Community |
|---|---|---|
| 先生 | Qwen2.5-14B-Instruct | `mlx-community/Qwen2.5-14B-Instruct-4bit` |
| 生徒 | Qwen2.5-1.5B-Instruct | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |

## 前提

- macOS + Apple Silicon（M1 以降）
- Python 3.12（`uv` 推奨）
- 空きメモリ 16GB 以上（14B 4bit + 学習用）

## セットアップ

```bash
git clone https://github.com/masanori0209/slm-humor-distill-demo.git
cd slm-humor-distill-demo

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 一括実行

```bash
bash scripts/run-all.sh
```

初回は Hugging Face から 14B / 1.5B を DL するため時間がかかります。

## 個別実行

```bash
# 1. 教師データ生成（先生 14B）
python scripts/generate_teacher_data.py

# 2. train/valid 分割
python scripts/prep_train_data.py

# 3. LoRA 学習
bash scripts/train_lora.sh

# 4. アダプタをベースにマージ
bash scripts/fuse_model.sh

# 5. before/after 比較
python scripts/eval_before_after.py --fused-model fused_model/humor-1.5b-fused
python scripts/bench_mlx.py --model fused_model/humor-1.5b-fused
```

## 構成

```text
slm-humor-distill-demo/
├── configs/
│   ├── models.yaml      # 先生・生徒の HF パス
│   └── lora.yaml        # LoRA ハイパラ
├── data/
│   ├── prompts/seed_topics.txt
│   └── train/           # prep 後の train.jsonl / valid.jsonl
├── scripts/
│   ├── generate_teacher_data.py
│   ├── prep_train_data.py
│   ├── train_lora.sh
│   ├── fuse_model.sh
│   ├── eval_before_after.py
│   ├── bench_mlx.py
│   └── run-all.sh
├── adapters/            # LoRA 出力（git 管理外）
├── fused_model/         # マージ後（git 管理外）
└── reports/             # 評価・ベンチログ（git 管理外）
```

## CPU のみで動かす場合

MLX は Apple Silicon の **Metal（GPU）** を使います。厳密な CPU-only は GGUF 変換 + `llama.cpp` の `-ngl 0` が必要です（記事側で手順を書く予定）。

## 参考

- [TinySwallow-1.5B / TAID](https://sakana.ai/taid-jp/)
- [mlx-lm](https://github.com/ml-explore/mlx-lm)

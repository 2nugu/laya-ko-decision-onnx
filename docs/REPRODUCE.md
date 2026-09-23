# Reproducing the evaluation and the fine-tune

## Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.12, PyTorch 2.9.1+cu128, transformers 4.57.6, one RTX PRO 6000
(Blackwell, 97 GB). The model is 322 M parameters, so any GPU with ≥ 16 GB is plenty;
training used 39 GB at batch 32 with gradient checkpointing off.

### Data locations

Defaults are repo-relative and a fresh clone needs no configuration: KLUE,
`LocalLLaMA/typed-decisions` and Kev `decision-v2` download on first use and cache under
`data/`. Override any path with an environment variable — see `.env.example`:

| variable | default | holds |
|---|---|---|
| `LAYA_KO_ROOT` | parent of `scripts/` | the repo |
| `LAYA_KO_DATA` | `$ROOT/data` | caches, built training set |
| `LAYA_KO_MODELS` | `$ROOT/models` | checkpoints, ONNX builds |
| `LAYA_KO_EVAL_DIR` | `$DATA/eval_datasets` | `kmmlu/` `mmlu/` `jcommonsenseqa/` |
| `LAYA_KO_AIHUB_DIR` | `$DATA/korean_datasets` | AI-Hub jsonl corpora |

The last two point at data this repository does not redistribute. Benchmarks that need
them print a `[skip]` line naming the missing path and are left out of the table; the
rest run normally. Concretely, a clone with no local corpora reproduces `klue_*`,
`en_typed_decisions` and `en_kev_v2`, and skips `ko_kmmlu`, `en_mmlu`, `ja_jcqa` and
`ko_culture`.

### A note on transformers < 5

The upstream Laya checkpoints store their encoder config in the transformers 5.0 schema
(`rope_parameters`). On transformers 4.x that key is read but not applied, so 16 of the
22 sliding-attention layers in the multilingual checkpoint silently run with rope base
10 000 instead of the 160 000 the checkpoint declares.

`scripts/bench.py::fix_rope` rebuilds the affected `inv_freq` buffers, and every script
here calls it. The measured effect is small (+0.004 overall accuracy) because the
sliding window is only 128 tokens, but it is a real mismatch, and the published
checkpoint bakes the corrected values into `encoder/config.json` so downstream loaders
get it right without the patch.

## Evaluate

```bash
# baseline (upstream multilingual checkpoint)
python scripts/bench.py --rope fixed --out eval/results/baseline.json \
  --benches klue_ynat,klue_nli,klue_sts,klue_re,ko_culture,ko_kmmlu,ja_jcqa,en_typed_decisions,en_kev_v2,en_mmlu

# this model
python scripts/bench.py --model <checkpoint-dir> --subfolder "" --out eval/results/after.json \
  --benches klue_ynat,klue_nli,klue_sts,klue_re,ko_culture,ko_kmmlu,ja_jcqa,en_typed_decisions,en_kev_v2,en_mmlu

python scripts/compare.py --before eval/results/baseline.json --after eval/results/after.json
```

KLUE, `LocalLLaMA/typed-decisions` and Kev `decision-v2` are fetched at run time. KMMLU,
MMLU and JCommonsenseQA are read from `$LAYA_KO_EVAL_DIR` as normalized jsonl (one object
per line, fields as in the original datasets), and `ko_culture` from `$LAYA_KO_AIHUB_DIR`;
all four are skipped if absent.

Each benchmark item becomes one typed decision: the question text goes in `state`, a short
task instruction in `instructions`, and the answer options in `criteria`. Multiple-choice
sets map to `choice`, KLUE-STS to `score` (the 0–5 similarity label binned to 6 ordinal
levels), and boolean items to `noul`.

### Harness validation

Before trusting any number here, check that the harness reproduces a known result: the
upstream `typed-decisions` checkpoint should score ≈ 0.769 (acc@50 % coverage ≈ 0.900)
on `en_typed_decisions`.

```bash
python scripts/bench.py --subfolder typed-decisions --benches en_typed_decisions
```

## Train

```bash
python scripts/traindata.py                    # writes data/train_items.jsonl
python scripts/train.py --epochs 3 --out models/laya-ko
```

`traindata.py` pulls the KLUE training splits and `LocalLLaMA/typed-decisions` over the
network, and reads AI-Hub corpora from a local path (see README § Training data). Without
those local files it still builds a usable KLUE + replay mixture; the AI-Hub families are
simply absent.

Defaults: encoder LR 1e-5, head LR 1e-4, OneCycle schedule, effective batch 64, EN replay
oversampled to ≈ 19 % of each epoch. The loss is the upstream RLCD objective — a policy
gradient on a strictly proper scoring rule (log + spherical, plus RPS for ordinal
questions) plus soft cross-entropy — carried over unchanged from the Laya fine-tuning
notebook. Training prints the full retention panel after every epoch, so forgetting shows
up while there is still time to stop.

Wall clock: about 5 minutes per epoch at 470 sequences/s on one GPU.

### Why the encoder LR is low

The recipe fully fine-tunes the encoder, not an adapter. At the notebook's 2.5e-5 the
Korean gains arrive slightly faster but the multilingual encoder drifts; 1e-5 plus
oversampled English replay keeps Japanese and English flat or improving. `--l2sp` adds an
L2 penalty toward the starting encoder weights if you want a harder constraint; it was not
needed here and defaults to off.

## Export to ONNX

See [ONNX.md](ONNX.md).

```bash
python scripts/export_onnx.py --model models/laya-ko --subfolder "" \
  --dynamo --opset 20 --logits-only --out models/onnx/model.onnx
python scripts/make_golden.py
python scripts/bench_onnx.py --onnx models/onnx/model.onnx --model-dir models/laya-ko
```

`bench_onnx.py` runs the same benchmarks through ONNX Runtime; its accuracies should match
the PyTorch run to within a few items.

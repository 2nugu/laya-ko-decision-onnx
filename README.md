# laya-ko

Korean fine-tune of [Laya](https://huggingface.co/convaiinnovations/laya), a small
non-generative decision model. Same API, same 322 M parameters, same single forward
pass — it just answers Korean typed decisions far better.

**Weights:** [`2nugu/laya-ko`](https://huggingface.co/) · **License:** Apache-2.0

> 라야(Laya)가 한국어를 잘 못하길래 튜닝해 보았습니다. 한국어 이용자분들 파이팅!

---

## What this model is (read this first)

Laya is **not** a generative LLM. It is a bidirectional encoder (mmBERT-base) with a
small decision head. Every option in a question gets its own `[MASK]` marker, the head
scores those markers, and the scores are softmaxed **over that question's options only**:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
                                              ↓         ↓
                                           score     score   →  softmax  →  probabilities
```

It emits no tokens. You hand it a piece of text plus a typed question, and it hands back
a calibrated probability distribution over the options you defined. Three question types:

| type | you give | you get |
|---|---|---|
| `choice` | named options, optionally described | the best option + a probability per option |
| `score` | ordered levels, low to high | the mean level + a distribution over levels |
| `noul` | a yes/no statement | the probability that it holds |

That makes it useful for classification, routing, intent detection, moderation gates,
relevance judging, and grading — the jobs usually handed to a 7B model with a JSON
prompt and a regex. It does those in ~30 ms on a CPU, with a probability you can
threshold on. It cannot summarize, translate, or chat.

The options are supplied at call time, so you can change the label set without
retraining. It does not need to have seen your labels during training.

## Results

Fine-tuning moved the Korean numbers a long way while leaving the other languages alone.
Accuracy, and ECE (expected calibration error — lower is better):

| benchmark | n | chance | before | **after** | Δ | ECE before → after |
|---|---|---|---|---|---|---|
| KLUE-RE (relation, 30-way) | 1000 | 0.033 | 0.136 | **0.705** | **+56.9 pp** | 0.184 → 0.065 |
| KLUE-YNAT (topic, 7-way) | 1000 | 0.143 | 0.414 | **0.834** | **+42.0 pp** | 0.315 → 0.077 |
| KLUE-STS (ordinal, 6 levels) | 519 | 0.167 | 0.210 | **0.509** | **+29.9 pp** | 0.120 → 0.188 |
| AI-Hub culture MC | 900 | 0.367 | 0.373 | **0.628** | **+25.4 pp** | 0.250 → 0.084 |
| KLUE-NLI (3-way) | 1000 | 0.333 | 0.761 | **0.815** | +5.4 pp | 0.144 → 0.113 |
| KMMLU (knowledge MC) | 900 | 0.250 | 0.244 | 0.298 | +5.3 pp | 0.203 → 0.253 |
| *typed-decisions (EN)* | 2000 | 0.318 | 0.350 | **0.725** | +37.5 pp | 0.319 → 0.243 |
| *JCommonsenseQA (JA)* | 500 | 0.200 | 0.526 | **0.566** | +4.0 pp | 0.025 → 0.084 |
| *Kev decision-v2 (EN)* | 1440 | 0.300 | 0.585 | 0.572 | −1.4 pp | 0.267 → 0.307 |
| *MMLU (EN)* | 560 | 0.250 | 0.295 | 0.266 | −2.9 pp | 0.140 → 0.297 |

*Italic rows are retention checks — they were never trained on and exist to catch
forgetting.* English and Japanese held: Japanese improved, and the only real English cost
is 1.4 points on Kev decision-v2. English typed-decisions rose because 19 % of each
training epoch was English replay drawn from that distribution.

Reproduce any row with `python scripts/bench.py` — see [docs/REPRODUCE.md](docs/REPRODUCE.md).
KLUE, typed-decisions and Kev decision-v2 download on first use; the KMMLU, MMLU,
JCommonsenseQA and AI-Hub rows need local corpora and are skipped with a message if you
do not have them.

### The baseline was not what it looked like

Worth stating plainly, because it changes what "Laya is bad at Korean" means: the
upstream multilingual checkpoint already solved KLUE-NLI at 0.761. Korean was not
broken. What was missing was ordinal judgement (STS at 0.210, barely above the 0.167
floor), high-cardinality classification (RE at 0.136), and calibration (YNAT at ECE
0.315 — confidently wrong). Those are what the fine-tune fixed.

Two benchmarks here are the wrong tool for this model and are reported only so nobody
reads them as a Korean signal: **KMMLU and MMLU are knowledge-recall tests**, and all
three upstream Laya checkpoints score 0.24–0.32 on MMLU regardless of language. A 322 M
encoder has no facts to recall. Their numbers move a little and mean little.

## Quick start

### Python

```bash
pip install laya
```

```python
import laya

agent = laya.load("2nugu/laya-ko")

state = "한국은행, 기준금리 0.25%p 인하 결정"
questions = {
    "topic": {
        "type": "choice",
        "instructions": "다음 뉴스 제목의 주제 분야를 고르세요.",
        "criteria": {
            "IT과학": "정보기술·과학 기사", "경제": "경제·금융·산업 기사",
            "사회": "사회·사건사고 기사", "생활문화": "생활·문화·연예 기사",
            "세계": "국제·해외 기사", "스포츠": "스포츠 기사", "정치": "정치·외교 기사",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "이 뉴스는 즉시 대응이 필요한 사안인가?",
    },
    "importance": {
        "type": "score",
        "instructions": "이 뉴스의 중요도를 판정하세요.",
        "criteria": ["매우 낮음", "낮음", "보통", "높음", "매우 높음"],
    },
}

result = agent.predict(state, questions)
print(result["answers"]["topic"]["choice"])         # 경제
print(result["answers"]["topic"]["probabilities"])  # {"경제": 1.0, "사회": 0.0, ...}
print(result["answers"]["urgent"]["noul"])          # 0.009
print(result["answers"]["importance"]["score"])     # 0.64  (mean level, 0–4)
```

All three questions go through in **one** forward pass. They share the `state` and cannot
see each other.

These are the actual outputs. Note what they show: `topic` is a label set the model was
trained on and it is decisive; `urgent` and `importance` are scales invented on the spot
that it has never seen, and it handles the boolean far better than the ordinal. That is
the general pattern — `choice` and `noul` transfer to new label sets much more readily
than `score` does.

### ONNX (any language)

```python
import onnxruntime as ort
sess = ort.InferenceSession("onnx/model.onnx")   # model.onnx.data must sit beside it
logits = sess.run(None, {"input_ids": ids, "attention_mask": att,
                         "marker_pos": pos, "marker_mask": mask, "qtype": qt})[0]
```

All axes are dynamic; one batch may mix question types and option counts. Measured on CPU
with ONNX Runtime: **28–48 ms per item** at batch 16, depending on option count.

### Rust

`rust/` is a reference client over the `ort` and `tokenizers` crates.

```bash
cd rust && cargo run --release --example classify -- /path/to/onnx-dir
```

> **Not compiled as published.** There was no Rust toolchain on the machine this was
> developed on, so `rust/` has not been built. The `ort` 2.0 release-candidate API is
> still moving — `Session::run`, `ort::inputs!` and `try_extract_tensor` have all changed
> signature between rc versions — so expect to adjust a few call sites for whichever rc
> you pin. The part that actually matters, `build_sequence`, is a line-by-line port of the
> Python and is verifiable against `onnx/golden.json` without touching the ONNX plumbing.

The one thing a non-Python client must reimplement is sequence construction — where the
`[MASK]` markers land, and how option text is truncated when it overflows the head budget.
That is `build_sequence` in `rust/src/lib.rs`, a direct port of the Python. `onnx/golden.json`
carries 13 cases (all three types, 2–40 options, Korean/Japanese/English, plus one
deliberately oversized case that exercises truncation) with the expected token ids, marker
positions and output probabilities, so a port can be verified rather than assumed.

See [docs/ONNX.md](docs/ONNX.md) for the full input contract, the export recipe, and the
precision numbers.

## Practical notes

**Write good option descriptions.** The option text is part of the input, and the model
scores it. `{"환불": "고객이 환불이나 취소를 요청함"}` works better than `{"환불": null}`.
This is the cheapest lever you have.

**Budget your options.** Instructions plus all options must fit in 256 tokens (each option
is capped at 48). Past roughly 30 medium-length options the code silently truncates every
option evenly; past that, descriptions stop helping. `state` gets the remaining room up to
1024 tokens and is truncated from the right.

**`score` is the weakest primitive.** It is the hardest of the three and improved the least
in relative terms. If an ordinal judgement can be re-expressed as a `choice`, it will
usually be more accurate.

**Recalibrate for your own data.** The shipped temperatures were fitted on a training
mixture that is ~81 % Korean, which is why calibration on English benchmarks is worse than
accuracy suggests. Refitting takes one forward pass over a few hundred labelled examples
and touches no weights:

```bash
python scripts/recalibrate.py --model <checkpoint-dir> --data my_dev.jsonl --dry-run
```

**Use the probabilities.** The model is meaningfully better on the half of inputs it is
most confident about — YNAT accuracy is 0.834 overall but 0.962 on the most-confident half.
If you can escalate uncertain cases to a larger model or a human, the threshold is where
most of the practical value is.

## Training

| | |
|---|---|
| base | `convaiinnovations/laya`, `multilingual` subfolder (mmBERT-base encoder, 322 M) |
| objective | upstream RLCD — policy gradient on a strictly proper scoring rule (log + spherical, plus RPS for ordinal) + soft cross-entropy |
| data | 108,665 typed decisions: 102,665 Korean + 6,000 English replay, oversampled ×4 to ≈ 19 % of each epoch |
| schedule | 4 epochs, effective batch 64, OneCycle; encoder LR 1e-5, head LR 1e-4 |
| hardware | one RTX PRO 6000 (Blackwell), **19 minutes**, 39 GB peak |

The encoder is fully fine-tuned — no adapter — so the low encoder LR and the English
replay are what keep the other languages from drifting. At the upstream notebook's 2.5e-5
the Korean gains come slightly faster and the multilingual encoder does not hold.

### Training data

Korean, from **KLUE** (CC BY-SA 4.0 — YNAT 20 k, NLI 15 k, STS 11.7 k, RE 15 k) and from
several **AI-Hub** corpora carrying usable metadata labels (culture MC, aspect sentiment,
news field, industry class, essay subject, and others). English replay is
`LocalLLaMA/typed-decisions` (Apache-2.0).

**The training set is not redistributed.** `scripts/traindata.py` rebuilds the KLUE and
replay portions over the network; the AI-Hub portions are read from local paths and are
simply absent if you do not have them, in which case the script still produces a usable
KLUE + replay mixture.

Evaluation splits are disjoint from training: KLUE training uses the `train` split and
evaluation the `validation` split. One caveat is worth stating — the AI-Hub culture
benchmark excludes its evaluation items by exact text match, but the same underlying
corpus is in the training mixture, so its +25.4 pp is in-domain transfer rather than a
clean zero-shot result. The KLUE rows do not have this problem.

## Limitations

- **Not generative.** No text out, ever. If you need a sentence, this is the wrong model.
- **No world knowledge.** KMMLU 0.298 against a 0.250 floor. It judges text in front of it;
  it does not know things.
- **Calibration is Korean-tuned.** See the recalibration note above.
- **Korean-first, not Korean-only, and not verified beyond three languages.** English and
  Japanese were checked. The base model covers 100+ languages; the other 97 were not
  measured and this fine-tune may well have cost them something.
- **The AI-Hub benchmark row is in-domain**, as described above.
- **Long inputs are truncated from the right** at 1024 tokens. There is no windowing.

## Attribution

Built on **Laya** by Convai Innovations (Apache-2.0) —
[weights](https://huggingface.co/convaiinnovations/laya) ·
[source](https://github.com/NandhaKishorM/laya) — which implements the decision-model
architecture that
[Kev](https://github.com/jaredpalmer/kev) also targets, both API-compatible with
TypeSafe's Jev. The training recipe follows Laya's own fine-tuning notebook, and `rust/`
ports `laya.common.build_sequence` and the post-processing in `laya.agent`, both from the
source repository above. The encoder is
[mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (Apache-2.0).

See [`NOTICE`](NOTICE) for the full attribution list, including every evaluation dataset
and its license.

### One upstream bug worth knowing about

The Laya checkpoints store their encoder config in the transformers 5.0 schema
(`rope_parameters`). On transformers 4.x that key is parsed but not applied, so 16 of the
22 sliding-attention layers in the multilingual checkpoint run with RoPE base 10,000
instead of the 160,000 the checkpoint declares — silently, with no warning.

Measured impact is small (+0.004 overall accuracy once corrected) because the sliding
window is only 128 tokens, but it is a genuine mismatch. `scripts/bench.py::fix_rope`
corrects it at load time, and this checkpoint's `encoder/config.json` has the values
written out explicitly, so loading it needs no patch.

## License

Apache-2.0, matching upstream Laya. Note that part of the training data is KLUE, which is
CC BY-SA 4.0; whether trained weights constitute an adaptation of a dataset under that
license is legally unsettled, and the common practice — followed here — is to release the
weights under the base model's license and document the data provenance. Check this
yourself if it matters for your use.

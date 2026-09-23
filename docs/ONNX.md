# ONNX export and non-Python deployment

The model is a single forward pass over a bidirectional encoder — no KV cache, no
autoregressive loop — so it exports to one ONNX graph and runs anywhere ONNX Runtime
runs. This document records the export recipe (four conditions, each of which was hit
as a real failure), the input contract, and how to verify a port.

## Export

```bash
python scripts/export_onnx.py \
  --model <checkpoint-dir> --subfolder "" \
  --dynamo --opset 20 --logits-only \
  --out onnx/model.onnx
```

Four things are required. Dropping any one of them fails, and two of them fail
*silently at a different input shape* rather than at export time:

| Requirement | What happens without it |
|---|---|
| `torch.backends.mha.set_fastpath_enabled(False)` | Export aborts: `aten::_transformer_encoder_layer_fwd` has no ONNX symbolic. The decision head is an `nn.TransformerEncoder`, whose nested-tensor fast path is not exportable. |
| `--dynamo` | Export *succeeds*, and is numerically exact at the traced shape — then throws `Reshape` errors at every other shape. The legacy tracer bakes the traced batch and sequence length into the head's `nn.MultiheadAttention`. |
| `--opset 20` | ONNX Runtime refuses the model: `Unrecognized attribute: num_outputs for operator Split`. `Split.num_outputs` is opset 18+, and the dynamo exporter emits it regardless of the requested opset. |
| `--logits-only` | Everything works until you try to quantize, which dies in shape inference: `Inferred shape and existing shape differ in dimension 0: (772) vs (256)`. The act/escalate head is the only place the graph concatenates a pooled vector with derived top-k features (768 + 4 = 772). Nothing in a typed decision reads that head. |

The exporter runs a parity check against PyTorch on four shapes *other than* the traced
one and prints `PARITY: PASS` only if every post-softmax probability matches within
`--atol` (default `1e-3`). Measured on this checkpoint: max |Δp| = 3.8e-5.

## Files to ship

| File | Size | Purpose |
|---|---|---|
| `model.onnx` | 2.2 MB | the graph |
| `model.onnx.data` | 1.2 GB | fp32 weights — must sit next to `model.onnx` |
| `tokenizer.json` | 33 MB | loads directly with the Rust `tokenizers` crate |
| `laya_runtime.json` | 1 KB | everything the graph does not carry: temperatures, special token ids, token budgets |
| `golden.json` | 64 KB | expected outputs for verifying a port |

## Input contract

| Name | Shape | Type | Meaning |
|---|---|---|---|
| `input_ids` | `[B, L]` | int64 | token ids from `build_sequence` |
| `attention_mask` | `[B, L]` | int64 | 1 for real tokens, 0 for padding |
| `marker_pos` | `[B, K]` | int64 | index of each option's `[MASK]` marker |
| `marker_mask` | `[B, K]` | bool | true for options this row actually has |
| `qtype` | `[B]` | int64 | 0 = choice, 1 = score, 2 = noul |

Output `logits [B, K]` float32, already filled with `-1e4` where `marker_mask` is false.

All axes are dynamic. `B` may mix question types and option counts in one batch; pad `K`
to the row with the most options and mark the rest false.

## Post-processing

```
bucket      = "{choice|score|noul}:{2 | 3-5 | 6-10 | 11+}"   by option count k
temperature = temperature_by_options[bucket]  or  temperature[qtype]
p           = softmax(logits[:k] / temperature)
```

Then read the answer by type: `choice` → `argmax(p)` maps to the criteria key at that
index; `noul` → `p[1]` is the probability of true; `score` → report the **mean level**
`Σ i·p[i]`, not the argmax.

## Sequence construction

This is the only part a non-Python client must reimplement. Layout:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
```

Rules, in order:

1. Render options in label order. `choice` → `"key: description"` (or just `"key"` when the
   description is null/empty); `score` → `"level {i}: {text}"`; `noul` → always exactly
   `["false: ...", "true: ..."]`, falling back to the default strings in `laya_runtime.json`.
2. Strip any literal `<mask>` from instructions, option text, and state — replace with a space.
3. Tokenize each option as `[MASK] + tokenize(" " + option_text)[:48]`. The leading space matters.
4. `opt_budget = head_max_len - Σ len(option)`. If that is below 16, truncate **every** option to
   `max(4, (head_max_len - 16) / n_options)` and recompute.
5. Truncate the instruction tokens to `max(8, opt_budget)`.
6. Assemble; record each option's marker position as it is appended.
7. Fill the remainder up to `max_len` with state tokens, then a final `[SEP]`.
8. If any marker landed at or past `max_len`, the question does not fit — reject it rather
   than scoring a truncated option set.

`rust/src/lib.rs` implements this. See `build_sequence`.

## Verifying a port

`golden.json` has 13 cases covering all three question types, 2–40 options, Korean /
Japanese / English input, and one deliberately oversized case that exercises the
truncation path at `max_len`. Each case carries the expected token count, the first 24
and last 8 token ids, a SHA-256 of the full id sequence, the marker positions, the
temperature that applies, and the final probabilities.

Check the token ids first. If they match, any remaining probability difference is
numerical and should be below `1e-4`; if they do not match, the bug is in steps 1–8
above and no amount of tolerance will hide it.

## Precision

Measured on CPU (ONNX Runtime, `CPUExecutionProvider`, batch 16):

| | size | klue_nli | klue_re | ja_jcqa | ms/item |
|---|---|---|---|---|---|
| fp32 | 1.2 GB | 0.815 | 0.703 | 0.556 | 28–48 |
| int8 dynamic | 310 MB | 0.724 | 0.617 | 0.500 | 22–39 |

**Naive int8 dynamic quantization is not worth it**: it costs up to 9 points of accuracy
to buy about 20 % throughput. If you need the model smaller, try per-channel weight
quantization with the embedding and LayerNorm nodes excluded, or quantization-aware
training — do not ship the default `quantize_dynamic` output.

On GPU, keep the fp32 graph and let the CUDA execution provider handle precision. The
fp16 file produced by `onnxconverter_common.float16` on this graph is broken — ONNX
Runtime rejects it with a type mismatch on a `Cast` node — so it is not published.

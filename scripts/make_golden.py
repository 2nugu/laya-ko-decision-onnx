"""Golden vectors so a Rust/other-language port can be verified against Python exactly."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchdata
import paths
from laya.common import QTYPES, build_sequence, render_options, temp_bucket

MODEL = os.environ.get("LAYA_KO_CKPT", paths.models("laya-ko"))
ONNX = os.environ.get("LAYA_KO_ONNX", paths.models("onnx", "model.onnx"))
OUT = paths.models("onnx", "golden.json")

import onnxruntime as ort
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.path.join(MODEL, "tokenizer"))
cfg = json.load(open(os.path.join(MODEL, "rl_agent_config.json")))
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

picks = []
for name in ("klue_ynat", "klue_nli", "klue_sts", "klue_re", "ja_jcqa", "en_kev_v2"):
    items = benchdata.BUILDERS[name]()
    picks += items[:2]
# a deliberately over-long option set, to exercise the shrink path
picks.append({"bench": "synthetic_overflow", "gold_idx": 0,
              "state": "상태 텍스트 " * 300,
              "q": {"t": "choice", "ins": "매우 " * 120 + "긴 지시문입니다.",
                    "crit": {f"opt{i}": "아주 긴 옵션 설명 " * 12 for i in range(40)}}})

cases = []
for it in picks:
    q = it["q"]
    seq, mk = build_sequence(tok, it["state"], q, cfg["max_len"], cfg["head_max_len"])
    if len(mk) != len(render_options(q)):
        continue
    qt = QTYPES[q["t"]]
    ids = np.array([seq], dtype=np.int64)
    att = np.ones_like(ids)
    pos = np.array([mk], dtype=np.int64)
    mm = np.ones((1, len(mk)), dtype=bool)
    logits = sess.run(None, {"input_ids": ids, "attention_mask": att, "marker_pos": pos,
                             "marker_mask": mm, "qtype": np.array([qt], dtype=np.int64)})[0]
    k = len(mk)
    t = cfg.get("temperature_by_options", {}).get(temp_bucket(qt, k), cfg["temperature"][qt])
    z = logits[0, :k].astype(np.float64) / t
    p = np.exp(z - z.max()); p = p / p.sum()
    crit = q["crit"]
    cases.append({
        "bench": it["bench"], "qtype": q["t"], "instructions": q["ins"],
        "criteria": (list(crit.items()) if isinstance(crit, dict) else [[c, None] for c in crit]),
        "state": it["state"],
        "expect": {"n_tokens": len(seq), "input_ids_head": seq[:24], "input_ids_tail": seq[-8:],
                   "input_ids_sha": __import__("hashlib").sha256(
                       json.dumps(seq).encode()).hexdigest()[:16],
                   "markers": mk, "temperature": float(t),
                   "probabilities": [round(float(v), 6) for v in p]},
    })

paths.ensure(OUT)
json.dump({"model": "laya-ko", "onnx": "model.onnx",
           "note": "probabilities produced by onnxruntime CPU fp32; compare with atol=1e-4",
           "cases": cases}, open(OUT, "w"), ensure_ascii=False, indent=2)
print(f"wrote {OUT}: {len(cases)} cases")
for c in cases[:4]:
    print(f"  {c['bench']:20s} {c['qtype']:6s} k={len(c['expect']['probabilities']):2d} "
          f"tokens={c['expect']['n_tokens']:4d} markers={c['expect']['markers'][:4]} "
          f"p0={c['expect']['probabilities'][0]:.4f}")
print(f"  {cases[-1]['bench']:20s} k={len(cases[-1]['expect']['probabilities'])} "
      f"tokens={cases[-1]['expect']['n_tokens']} (shrink path)")

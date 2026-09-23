"""End-to-end check: run the benchmark through onnxruntime instead of PyTorch."""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchdata
from bench import summarize
from laya.common import QTYPES, build_sequence, render_options, temp_bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--model-dir", required=True, help="dir with tokenizer/ and rl_agent_config.json")
    ap.add_argument("--benches", default="klue_ynat,klue_nli,klue_re,klue_sts,ja_jcqa,en_kev_v2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--providers", default="CPUExecutionProvider")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(a.model_dir, "tokenizer"))
    cfg = json.load(open(os.path.join(a.model_dir, "rl_agent_config.json")))
    temps = cfg.get("temperature", [1.0, 1.0, 1.0])
    by_opt = cfg.get("temperature_by_options", {})

    so = ort.SessionOptions()
    sess = ort.InferenceSession(a.onnx, so, providers=a.providers.split(","))
    print(f"[onnx] {a.onnx} providers={sess.get_providers()}")

    recs, lat = [], []
    for name in a.benches.split(","):
        items = benchdata.BUILDERS[name]()
        built = []
        for it in items:
            q = it["q"]
            seq, mk = build_sequence(tok, it["state"], q, cfg["max_len"], cfg["head_max_len"])
            if len(mk) != len(render_options(q)):
                continue
            built.append((seq, mk, QTYPES[q["t"]], it["gold_idx"], it["bench"]))
        built.sort(key=lambda x: len(x[0]))
        t0 = time.time()
        for s in range(0, len(built), a.batch_size):
            ch = built[s:s + a.batch_size]
            L = max(len(c[0]) for c in ch)
            K = max(len(c[1]) for c in ch)
            ids = np.full((len(ch), L), tok.pad_token_id, dtype=np.int64)
            att = np.zeros((len(ch), L), dtype=np.int64)
            pos = np.zeros((len(ch), K), dtype=np.int64)
            mm = np.zeros((len(ch), K), dtype=bool)
            qt = np.array([c[2] for c in ch], dtype=np.int64)
            for i, (seq, mk, _, _, _) in enumerate(ch):
                ids[i, :len(seq)] = seq
                att[i, :len(seq)] = 1
                pos[i, :len(mk)] = mk
                mm[i, :len(mk)] = True
            outs = sess.run(None, {"input_ids": ids, "attention_mask": att,
                                   "marker_pos": pos, "marker_mask": mm, "qtype": qt})
            logits = outs[0]
            for i, (seq, mk, qtv, gold, bench) in enumerate(ch):
                k = len(mk)
                z = logits[i, :k].astype(np.float64) / by_opt.get(temp_bucket(qtv, k), temps[qtv])
                p = np.exp(z - z.max())
                p = p / p.sum()
                recs.append({"bench": bench, "qtype": qtv, "k": k, "p": p, "gold": gold,
                             "pred": int(p.argmax()), "conf": float(p.max())})
        dt = time.time() - t0
        lat.append((name, len(built), 1000 * dt / max(1, len(built))))
        print(f"    {name}: {len(built)} items, {1000*dt/max(1,len(built)):.1f} ms/item")

    res = summarize(recs)
    print(f"\n{'bench':22s} {'n':>5s} {'acc':>7s} {'ECE':>7s}")
    for b, m in res.items():
        print(f"{b:22s} {m['n']:5d} {m['acc']:7.3f} {m['ece']:7.3f}")
    if a.out:
        json.dump({"onnx": a.onnx, "results": res,
                   "latency_ms_per_item": {n: round(v, 2) for n, _, v in lat}},
                  open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

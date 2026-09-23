"""Batched before/after benchmark for Laya (multilingual) on KO / EN / JA typed decisions."""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchdata
from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket, ece_score


def fix_rope(model, verbose=True):
    """transformers<5 ignores `rope_parameters`; sliding layers silently fall back to theta=10000.

    Rebuild each layer's inv_freq from the theta the checkpoint actually declares.
    Returns the number of layers corrected.
    """
    cfg = model.encoder.config
    rp = getattr(cfg, "rope_parameters", None)
    if not isinstance(rp, dict):
        return 0
    want = {False: float(rp.get("full_attention", {}).get("rope_theta", cfg.global_rope_theta)),
            True: float(rp.get("sliding_attention", {}).get("rope_theta", cfg.local_rope_theta))}
    fixed = 0
    for i, layer in enumerate(model.encoder.layers):
        attn = layer.attn
        rot = getattr(attn, "rotary_emb", None)
        if rot is None:
            continue
        is_local = tuple(attn.local_attention) != (-1, -1)
        base = want[is_local]
        f = rot.inv_freq
        d = f.numel() * 2
        cur = float(f[-1].double().pow(-d / (d - 2.0)).item())
        if abs(cur - base) / base < 1e-3:
            continue
        new = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32, device=f.device) / d))
        rot.inv_freq.copy_(new.to(f.dtype))
        if hasattr(rot, "original_inv_freq"):
            rot.original_inv_freq = rot.inv_freq.clone()
        fixed += 1
        if verbose and fixed <= 2:
            print(f"    layer {i}: rope base {cur:,.0f} -> {base:,.0f}")
    return fixed


@torch.no_grad()
def run(agent, items, batch_size=32, device="cuda:0", desc=""):
    tok, cfg, model = agent.tok, agent.cfg, agent.model
    temps = cfg.get("temperature", [1.0, 1.0, 1.0])
    temps_by_opt = cfg.get("temperature_by_options", {})
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    built = []
    for it in items:
        q = it["q"]
        seq, markers = build_sequence(tok, it["state"], q, cfg["max_len"], cfg["head_max_len"])
        k = len(render_options(q))
        if len(markers) != k:          # options did not fit in head_max_len
            continue
        built.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]],
                      "target": [0.0] * k, "label": it["gold_idx"], "episode": 0,
                      "ep_step": 0, "ep_len": 1, "src": "bench", "_bench": it["bench"]})
    dropped = len(items) - len(built)
    order = sorted(range(len(built)), key=lambda i: len(built[i]["ids"]))

    recs = []
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        chunk = [built[i] for i in order[s:s + batch_size]]
        b = collate_items([chunk], tok.pad_token_id)
        with torch.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["marker_pos"].to(device), b["marker_mask"].to(device),
                              b["qtype"].to(device))
        logits = logits.float().cpu().numpy()
        for r, it in enumerate(chunk):
            k = len(it["markers"])
            qt = it["qtype"]
            z = logits[r, :k] / temps_by_opt.get(temp_bucket(qt, k), temps[qt])
            p = np.exp(z - z.max())
            p = p / p.sum()
            recs.append({"bench": it["_bench"], "qtype": qt, "k": k, "p": p,
                         "gold": it["label"], "pred": int(p.argmax()), "conf": float(p.max())})
    dt = time.time() - t0
    print(f"    {desc}: {len(recs)} items in {dt:.1f}s ({1000*dt/max(1,len(recs)):.1f} ms/item)"
          + (f"  [dropped {dropped}]" if dropped else ""))
    return recs


def summarize(recs):
    out = {}
    benches = sorted({r["bench"] for r in recs}) + ["OVERALL"]
    for b in benches:
        sel = recs if b == "OVERALL" else [r for r in recs if r["bench"] == b]
        if not sel:
            continue
        corr = np.array([float(r["pred"] == r["gold"]) for r in sel])
        conf = np.array([r["conf"] for r in sel])
        brier, nll = [], []
        for r in sel:
            oh = np.zeros(r["k"]); oh[r["gold"]] = 1.0
            brier.append(float(((r["p"] - oh) ** 2).sum()))
            nll.append(-float(np.log(max(r["p"][r["gold"]], 1e-12))))
        # accuracy at 50% coverage (most-confident half)
        idx = np.argsort(-conf)[:max(1, len(sel) // 2)]
        out[b] = {"n": len(sel), "acc": float(corr.mean()), "ece": ece_score(conf, corr),
                  "brier": float(np.mean(brier)), "nll": float(np.mean(nll)),
                  "acc@50cov": float(corr[idx].mean()),
                  "chance": float(np.mean([1.0 / r["k"] for r in sel]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default="multilingual")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rope", choices=["asis", "fixed"], default="fixed")
    ap.add_argument("--benches", default="ko_kmmlu,ko_culture,en_mmlu,ja_jcqa,en_typed_decisions")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import laya
    print(f"[load] {a.model} subfolder={a.subfolder} rope={a.rope}")
    agent = laya.load(a.model, subfolder=a.subfolder or None, device=a.device)
    if a.rope == "fixed":
        n = fix_rope(agent.model)
        print(f"    corrected {n} layers")
    agent.model.eval()

    all_recs = []
    for name in a.benches.split(","):
        items = benchdata.BUILDERS[name]()
        if not items:
            continue
        all_recs += run(agent, items, a.batch_size, a.device, desc=name)
    if not all_recs:
        raise SystemExit("no benchmark produced any item — check the paths printed above")

    res = summarize(all_recs)
    print()
    hdr = f"{'bench':22s} {'n':>5s} {'acc':>7s} {'chance':>7s} {'ECE':>7s} {'Brier':>7s} {'NLL':>7s} {'acc@50':>7s}"
    print(hdr); print("-" * len(hdr))
    for b, m in res.items():
        print(f"{b:22s} {m['n']:5d} {m['acc']:7.3f} {m['chance']:7.3f} {m['ece']:7.3f} "
              f"{m['brier']:7.3f} {m['nll']:7.3f} {m['acc@50cov']:7.3f}")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"model": a.model, "subfolder": a.subfolder, "rope": a.rope,
                       "tag": a.tag, "results": res}, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

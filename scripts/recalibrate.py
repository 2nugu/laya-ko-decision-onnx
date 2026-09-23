"""Refit the decision temperatures on your own labelled data.

The published temperatures were fitted on this project's training mixture, which is
~81 % Korean. They are right for Korean classification and slightly off for anything
else — a model that is well calibrated on one distribution generally is not on another.
Refitting costs one forward pass over a few hundred labelled examples and changes no
weights; it only rewrites `temperature` and `temperature_by_options` in
`rl_agent_config.json`.

Input: a JSONL file, one typed decision per line, same shape the benchmarks use:

    {"state": "...",
     "q": {"t": "choice", "ins": "...", "crit": {"A": "desc or null", "B": null}},
     "gold_idx": 0}

`t` is one of choice / score / noul. For score, `crit` is a list of level texts in
ascending order. For noul, `crit` may be {} or {"false": "...", "true": "..."}.

    python scripts/recalibrate.py --model models/laya-ko --data my_dev.jsonl
"""
import argparse, json, os, shutil, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import fix_rope
from train import calibrate, tokenize_items
from laya.common import ece_score, temp_bucket


def report(agent, items, device, dtype, temps, by_opt, label):
    """Accuracy and ECE under a given temperature set, for before/after comparison."""
    from train import collate, length_batches
    tok = agent.tok
    conf, corr = [], []
    agent.model.eval()
    with torch.no_grad():
        for bidx in length_batches(items, 64, seed=0):
            chunk = [items[i] for i in bidx]
            b = collate(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
                lg, _ = agent.model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                    b["marker_pos"].to(device), b["marker_mask"].to(device),
                                    b["qtype"].to(device))
            lg = lg.float().cpu().numpy()
            for r, it in enumerate(chunk):
                k = len(it["markers"])
                qt = it["qtype"]
                t = by_opt.get(temp_bucket(qt, k), temps[qt])
                z = lg[r, :k].astype(np.float64) / t
                p = np.exp(z - z.max())
                p = p / p.sum()
                conf.append(float(p.max()))
                corr.append(float(int(p.argmax()) == it["label"]))
    conf, corr = np.array(conf), np.array(corr)
    print(f"    {label:10s} acc={corr.mean():.3f}  ECE={ece_score(conf, corr):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir (rewritten in place)")
    ap.add_argument("--data", required=True, help="JSONL of labelled typed decisions")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-items", type=int, default=5000)
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write")
    a = ap.parse_args()

    import laya
    agent = laya.load(a.model, device=a.device)
    fix_rope(agent.model, verbose=False)
    cfg = dict(agent.cfg)
    dtype = torch.bfloat16 if a.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16

    raw = [json.loads(l) for l in open(a.data)][: a.max_items]
    for r in raw:
        r.setdefault("lang", "xx")
        r.setdefault("family", "user")
    items, dropped = tokenize_items(agent.tok, raw, cfg["max_len"], cfg["head_max_len"])
    if len(items) < 50:
        raise SystemExit(f"need at least 50 usable items, got {len(items)} ({dropped} dropped)")
    print(f"[data] {len(items)} usable items ({dropped} dropped: options exceeded head_max_len)")

    old_t = cfg.get("temperature", [1.0, 1.0, 1.0])
    old_b = cfg.get("temperature_by_options", {})
    report(agent, items, a.device, dtype, old_t, old_b, "published")

    temps, by_opt = calibrate(agent.model, items, agent.tok, a.device, dtype)
    print(f"[fit]  temperature      {[round(t, 3) for t in old_t]} -> {[round(t, 3) for t in temps]}")
    for k in sorted(by_opt):
        was = f"{old_b[k]:.3f}" if k in old_b else "  -  "
        print(f"       {k:12s} {was} -> {by_opt[k]:.3f}")
    missing = sorted(set(old_b) - set(by_opt))
    if missing:
        print(f"       per-bucket fit skipped for {', '.join(missing)} "
              f"(fewer than 50 items each); those fall back to the per-type temperature.")
        print("       Supply more data covering those option counts if you need them fitted.")
    report(agent, items, a.device, dtype, temps, by_opt, "refitted")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    path = os.path.join(a.model, "rl_agent_config.json")
    shutil.copy(path, path + ".bak")
    cfg["temperature"] = temps
    cfg["temperature_by_options"] = by_opt
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path} (previous kept as rl_agent_config.json.bak)")
    print("If you serve the ONNX build, copy the same values into onnx/laya_runtime.json.")


if __name__ == "__main__":
    main()

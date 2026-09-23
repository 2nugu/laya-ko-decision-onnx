"""Fine-tune Laya-multilingual on Korean typed decisions, with EN replay for retention.

Recipe follows the upstream RLCD notebook (policy gradient on a strictly proper scoring rule
+ soft CE), with three changes for this setting:
  * lower encoder LR and optional L2-SP toward the base weights (anti-forgetting)
  * EN replay oversampled to a target share of each epoch
  * rope_parameters honoured on transformers<5 (see bench.fix_rope)
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import fix_rope, run as bench_run, summarize
import benchdata
import paths
from laya.common import QTYPES, build_sequence, proper_reward, render_options, temp_bucket


# --------------------------------------------------------------------------- data
def tokenize_items(tok, raw, max_len, head_max_len):
    built, dropped = [], 0
    for it in raw:
        q = it["q"]
        k = len(render_options(q))
        seq, markers = build_sequence(tok, it["state"], q, max_len, head_max_len)
        if len(markers) != k:
            dropped += 1
            continue
        tgt = [0.0] * k
        tgt[it["gold_idx"]] = 1.0
        built.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]],
                      "target": tgt, "label": it["gold_idx"], "lang": it["lang"],
                      "family": it["family"]})
    return built, dropped


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    tgt = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        m = len(it["ids"])
        ids[i, :m] = torch.tensor(it["ids"])
        att[i, :m] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        tgt[i, :len(it["target"])] = torch.tensor(it["target"])
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": tgt, "qtype": torch.tensor([it["qtype"] for it in items])}


def length_batches(items, micro_batch, seed):
    """Sort by length into buckets (throughput), then shuffle bucket order (gradient noise)."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    batches = [order[s:s + micro_batch] for s in range(0, len(order), micro_batch)]
    random.Random(seed).shuffle(batches)
    return batches


# --------------------------------------------------------------------------- calibration
def fit_temp(pairs):
    if len(pairs) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), kmax), -1e4)
    T = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


@torch.no_grad()
def calibrate(model, items, tok, device, dtype, micro_batch=64):
    model.eval()
    preds = []
    for bidx in length_batches(items, micro_batch, seed=0):
        chunk = [items[i] for i in bidx]
        b = collate(chunk, tok.pad_token_id)
        with torch.autocast("cuda", dtype=dtype):
            lg, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                          b["marker_pos"].to(device), b["marker_mask"].to(device),
                          b["qtype"].to(device))
        lg = lg.float().cpu().numpy()
        for r, it in enumerate(chunk):
            k = len(it["markers"])
            preds.append((it["qtype"], k, lg[r, :k], it["target"]))
    temps = [1.0, 1.0, 1.0]
    for qt in range(3):
        sel = [(z, t) for q, k, z, t in preds if q == qt]
        if sel:
            temps[qt] = fit_temp(sel)
    by_opt = {}
    buckets = {}
    for q, k, z, t in preds:
        buckets.setdefault(temp_bucket(q, k), []).append((z, t))
    for b, sel in buckets.items():
        if len(sel) >= 50:
            by_opt[b] = fit_temp(sel)
    model.train()
    return temps, by_opt


# --------------------------------------------------------------------------- save
def save_ckpt(out_dir, model, tok, cfg, temps, by_opt, meta):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    enc_cfg = model.encoder.config
    # persist the corrected rope so the checkpoint is right under transformers<5 too
    rp = getattr(enc_cfg, "rope_parameters", None)
    if isinstance(rp, dict):
        enc_cfg.global_rope_theta = float(rp.get("full_attention", {}).get("rope_theta", enc_cfg.global_rope_theta))
        enc_cfg.local_rope_theta = float(rp.get("sliding_attention", {}).get("rope_theta", enc_cfg.local_rope_theta))
    enc_cfg.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    c = dict(cfg)
    c.update({"fine_tuned": True, "model_name": "laya-ko", "temperature": temps,
              "temperature_by_options": by_opt, "training": meta})
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(c, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=paths.data("train_items.jsonl"))
    ap.add_argument("--out", default=paths.models("laya-ko"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--replay-share", type=float, default=0.20, help="target EN share per epoch")
    ap.add_argument("--l2sp", type=float, default=0.0, help="L2 penalty toward base encoder weights")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--rl-weight", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs (debug knob)")
    ap.add_argument("--eval-benches", default="klue_ynat,klue_nli,klue_sts,klue_re,ja_jcqa,en_typed_decisions,en_mmlu")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = a.device
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    import laya
    print(f"[load] base = convaiinnovations/laya:multilingual")
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device=device)
    print(f"    rope: corrected {fix_rope(agent.model, verbose=False)} layers")
    model, tok, cfg = agent.model, agent.tok, dict(agent.cfg)

    base_enc = None
    if a.l2sp > 0:
        base_enc = {n: p.detach().clone() for n, p in model.encoder.named_parameters()}

    raw = [json.loads(l) for l in open(a.train)]
    items, dropped = tokenize_items(tok, raw, cfg["max_len"], cfg["head_max_len"])
    ko = [i for i in items if i["lang"] == "ko"]
    en = [i for i in items if i["lang"] != "ko"]
    print(f"[data] {len(items)} usable ({dropped} dropped: options exceeded head_max_len) "
          f"| ko={len(ko)} en={len(en)}")
    reps = max(1, round(a.replay_share * len(ko) / max(1, (1 - a.replay_share) * len(en))))
    epoch_items = ko + en * reps
    print(f"[data] EN replay repeated x{reps} -> epoch size {len(epoch_items)} "
          f"(EN share {len(en)*reps/len(epoch_items):.1%})")

    model.encoder.gradient_checkpointing_disable()
    model.head_checkpointing = False
    model.to(device).train()

    enc_p = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_p = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr_encoder},
                             {"params": head_p, "lr": a.lr_head}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(epoch_items) / (a.micro_batch * a.grad_accum))
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr_encoder, a.lr_head], total_steps=max(1, total), pct_start=0.06,
        anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)

    hist = []
    t_all = time.time()
    for ep in range(a.epochs):
        batches = length_batches(epoch_items, a.micro_batch, seed=a.seed + ep)
        sigma = a.sigma_start + (a.sigma_end - a.sigma_start) * (ep / max(1, a.epochs - 1))
        run_loss, nb, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for bi, bidx in enumerate(batches):
            if a.max_steps and nb >= a.max_steps:
                break
            chunk = [epoch_items[i] for i in bidx]
            b = collate(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=dtype):
                logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                    b["marker_pos"].to(device), b["marker_mask"].to(device),
                                    b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            target = b["target"].to(device)
            qtype = b["qtype"].to(device)
            k = mask.sum(-1, keepdim=True).float()

            eps = torch.randn((a.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            qd = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(qd, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = a.rl_weight * loss_rl + a.ce_weight * loss_ce
            if base_enc is not None:
                reg = sum(((p - base_enc[n]) ** 2).sum() for n, p in model.encoder.named_parameters())
                loss = loss + a.l2sp * reg
            (loss / a.grad_accum + 0.0 * act.sum()).backward()

            if (bi + 1) % a.grad_accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += float(loss)
            nb += 1
            if nb % 200 == 0:
                el = time.time() - t0
                print(f"  ep{ep+1} {nb}/{len(batches)} loss={run_loss/nb:.4f} "
                      f"ce={float(loss_ce):.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"{nb*a.micro_batch/el:.0f} seq/s", flush=True)

        print(f"=== epoch {ep+1}/{a.epochs} done in {time.time()-t0:.0f}s "
              f"avg_loss={run_loss/max(1,nb):.4f} ===", flush=True)

        calib = random.Random(0).sample(epoch_items, min(3000, len(epoch_items)))
        temps, by_opt = calibrate(model, calib, tok, device, dtype)
        print(f"    temperatures {[round(t,3) for t in temps]} | by_options {len(by_opt)} buckets")
        agent.cfg["temperature"] = temps
        agent.cfg["temperature_by_options"] = by_opt
        recs = []
        for name in a.eval_benches.split(","):
            recs += bench_run(agent, benchdata.BUILDERS[name](), 64, device, desc=f"ep{ep+1}:{name}")
        res = summarize(recs)
        hist.append({"epoch": ep + 1, "avg_loss": run_loss / max(1, nb), "results": res})
        print(f"    {'bench':22s} {'acc':>7s} {'ECE':>7s}")
        for bname, m in res.items():
            print(f"    {bname:22s} {m['acc']:7.3f} {m['ece']:7.3f}")
        model.train()

        save_ckpt(a.out, model, tok, cfg, temps, by_opt,
                  {"epochs_completed": ep + 1, "total_epochs": a.epochs,
                   "hours": (time.time() - t_all) / 3600, "base": "convaiinnovations/laya:multilingual",
                   "args": vars(a)})
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=2, ensure_ascii=False)
        print(f"    saved -> {a.out}", flush=True)

    print(f"\nTotal {(time.time()-t_all)/3600:.2f} h")


if __name__ == "__main__":
    main()

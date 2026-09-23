"""Export a Laya decision model to a single ONNX graph for Rust (`ort`) deployment.

Inputs  : input_ids[B,L] int64, attention_mask[B,L] int64, marker_pos[B,K] int64,
          marker_mask[B,K] bool, qtype[B] int64
Outputs : logits[B,K] float32 (already -1e4 masked), act_logits[B,2] float32

Post-processing in Rust (mirrors laya's rl_agent_api.system_one):
    z = logits[:k] / temperature_by_options.get(bucket(qtype,k), temperature[qtype])
    p = softmax(z)
Sequence construction (build_sequence) must be ported to Rust; tokenizer/tokenizer.json
loads directly with the `tokenizers` crate.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import fix_rope
import paths


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)


class LogitsOnlyWrapper(torch.nn.Module):
    """DecisionModel.forward up to `logits`, dropping the act/escalate head.

    The act head is the only place the graph mixes a pooled vector with derived
    top-k features (768+4 -> 772), which trips ONNX shape inference and blocks
    quantization; nothing in a typed decision reads it.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m = self.m
        h = m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]
        if m.head is not None:
            pad = ~attention_mask.bool()
            for layer in m.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        g = torch.gather(h, 1, idx)
        logits = m.scorer(g).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


def make_dummy(B, L, K, vocab, device):
    ids = torch.randint(5, min(vocab, 30000), (B, L), dtype=torch.long, device=device)
    att = torch.ones((B, L), dtype=torch.long, device=device)
    att[-1, L // 2:] = 0                                   # exercise the padding path
    pos = torch.randint(1, L // 2, (B, K), dtype=torch.long, device=device)
    mmask = torch.ones((B, K), dtype=torch.bool, device=device)
    mmask[-1, -1] = False                                  # exercise the option-mask path
    qt = torch.randint(0, 3, (B,), dtype=torch.long, device=device)
    return ids, att, pos, mmask, qt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default="multilingual")
    ap.add_argument("--out", default=paths.models("onnx", "model.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--dynamo", action="store_true")
    ap.add_argument("--logits-only", action="store_true",
                    help="drop the act/escalate head (unblocks shape inference + quantization)")
    ap.add_argument("--atol", type=float, default=1e-3, help="tolerance on post-softmax probabilities")
    a = ap.parse_args()

    # nn.TransformerEncoderLayer's nested-tensor fast path lowers to
    # aten::_transformer_encoder_layer_fwd, which has no ONNX symbolic. Force the
    # plain path so the head exports as ordinary matmul/softmax ops.
    torch.backends.mha.set_fastpath_enabled(False)

    import laya
    agent = laya.load(a.model, subfolder=a.subfolder or None, device=a.device)
    fix_rope(agent.model, verbose=False)
    model = agent.model.eval().float()
    model.encoder.config.reference_compile = False
    model.encoder.config._attn_implementation = "sdpa"
    if model.head is not None:
        model.head.enable_nested_tensor = False
        for lyr in model.head.layers:
            lyr.self_attn.batch_first = True
    wrapper = (LogitsOnlyWrapper(model) if a.logits_only else ExportWrapper(model)).eval()

    vocab = model.encoder.config.vocab_size
    dummy = make_dummy(2, 64, 4, vocab, a.device)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    names_in = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
    names_out = ["logits"] if a.logits_only else ["logits", "act_logits"]
    dyn = {"input_ids": {0: "B", 1: "L"}, "attention_mask": {0: "B", 1: "L"},
           "marker_pos": {0: "B", 1: "K"}, "marker_mask": {0: "B", 1: "K"},
           "qtype": {0: "B"}, "logits": {0: "B", 1: "K"}}
    if not a.logits_only:
        dyn["act_logits"] = {0: "B"}

    print(f"[export] opset={a.opset} dynamo={a.dynamo} -> {a.out}")
    with torch.no_grad():
        torch.onnx.export(wrapper, dummy, a.out, input_names=names_in, output_names=names_out,
                          dynamic_axes=dyn, opset_version=a.opset, do_constant_folding=True,
                          dynamo=a.dynamo)
    print(f"    wrote {os.path.getsize(a.out)/1e6:.1f} MB")

    # ---- parity check against PyTorch on a *different* shape than the trace shape
    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    print("    ort inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    ok = True
    for (B, L, K) in [(2, 64, 4), (1, 200, 2), (3, 128, 7), (1, 512, 30)]:
        d = make_dummy(B, L, K, vocab, a.device)
        with torch.no_grad():
            ref = wrapper(*d)
        if torch.is_tensor(ref):
            ref = (ref,)
        feed = {"input_ids": d[0].cpu().numpy(), "attention_mask": d[1].cpu().numpy(),
                "marker_pos": d[2].cpu().numpy(), "marker_mask": d[3].cpu().numpy(),
                "qtype": d[4].cpu().numpy()}
        got = sess.run(None, feed)

        def sm(x):
            x = x.astype(np.float64)
            x = x - x.max(-1, keepdims=True)
            e = np.exp(x)
            return e / e.sum(-1, keepdims=True)

        for nm, r, g in zip(names_out, ref, got):
            r = r.detach().cpu().numpy()
            dl = float(np.abs(r.astype(np.float64) - g.astype(np.float64)).max())
            dp = float(np.abs(sm(r) - sm(g)).max())     # what actually reaches the caller
            flag = "ok " if dp <= a.atol else "FAIL"
            ok &= dp <= a.atol
            print(f"    B={B:<2d} L={L:<4d} K={K:<3d} {nm:11s} "
                  f"max|Δlogit| = {dl:.2e}  max|Δprob| = {dp:.2e}  {flag}")
    print("\nPARITY:", "PASS" if ok else "FAIL")

    if a.fp16:
        from onnxconverter_common import float16
        import onnx
        m = onnx.load(a.out)
        m16 = float16.convert_float_to_float16(m, keep_io_types=True)
        p16 = a.out.replace(".onnx", ".fp16.onnx")
        onnx.save(m16, p16)
        print(f"    fp16 -> {p16} ({os.path.getsize(p16)/1e6:.1f} MB)")

    # ---- runtime sidecar: everything Rust needs that is not in the graph
    side = {"max_len": agent.cfg["max_len"], "head_max_len": agent.cfg["head_max_len"],
            "temperature": agent.cfg.get("temperature", [1.0, 1.0, 1.0]),
            "temperature_by_options": agent.cfg.get("temperature_by_options", {}),
            "qtypes": {"choice": 0, "score": 1, "noul": 2},
            "special_tokens": {"cls": agent.tok.cls_token_id, "sep": agent.tok.sep_token_id,
                               "mask": agent.tok.mask_token_id, "pad": agent.tok.pad_token_id},
            "mask_token": agent.tok.mask_token,
            "option_token_cap": 48, "head_min_tokens": 8, "opt_budget_floor": 16,
            "opt_per_option_floor": 4, "vocab_size": vocab,
            "head_prefix": "{qtype} question: {instructions}",
            "noul_default_false": "no, the statement does not hold",
            "noul_default_true": "yes, the statement holds",
            "logits_only": bool(a.logits_only), "opset": a.opset}
    sp = os.path.join(os.path.dirname(a.out), "laya_runtime.json")
    with open(sp, "w") as f:
        json.dump(side, f, indent=2)
    print(f"    sidecar -> {sp}")


if __name__ == "__main__":
    main()

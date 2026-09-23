"""Build the KO fine-tuning set + EN replay set as typed-decision items.

Output: list of {"family","lang","state","q":{t,ins,crit},"gold_idx"} — same shape as benchdata,
so the same tokenizer path (build_sequence) serves training and evaluation.

Eval leakage policy: KLUE *train* splits only (eval uses validation). AI-Hub 066 uses a
disjoint index set from the one benchdata.ko_culture samples.
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchdata as B
import paths

KD = paths.AIHUB_DIR


def _choice(family, lang, state, ins, crit, gold_idx):
    return {"family": family, "lang": lang, "state": state, "gold_idx": gold_idx,
            "q": {"t": "choice", "ins": ins, "crit": crit}}


# ---------------------------------------------------------------- KLUE train splits
def klue_train(caps=None):
    caps = caps or {"ynat": 20000, "nli": 15000, "sts": 11668, "re": 15000}
    out = []
    for cfg, cap in caps.items():
        fn = {"ynat": B.klue_ynat, "nli": B.klue_nli, "sts": B.klue_sts, "re": B.klue_re}[cfg]
        items = fn(n=cap, split="train", seed=17)
        for it in items:
            out.append({"family": f"klue_{cfg}", "lang": "ko", "state": it["state"],
                        "q": it["q"], "gold_idx": it["gold_idx"]})
    return out


# ---------------------------------------------------------------- AI-Hub 066 (train-disjoint)
def aihub_066(per_task=1200, seed=17):
    held = {(i["state"]) for i in B.ko_culture()}       # exact states used by the eval sample
    items = [it for it in B.ko_culture(per_task=10 ** 9, seed=seed) if it["state"] not in held]
    random.Random(seed).shuffle(items)
    by = {}
    out = []
    for it in items:
        k = it["q"]["t"] + str(len(it["q"]["crit"]))
        by[k] = by.get(k, 0) + 1
        if by[k] > per_task * 3:
            continue
        out.append({"family": "aihub066_culture", "lang": "ko", "state": it["state"],
                    "q": it["q"], "gold_idx": it["gold_idx"]})
    return out


# ---------------------------------------------------------------- AI-Hub 147 aspect sentiment
_POL = {"score": ["매우 부정적", "부정적", "중립적", "긍정적", "매우 긍정적"]}


def aihub_147(cap=12000, seed=17):
    rows = []
    for line in open(f"{KD}/aihub-147-aspect-sentiment.jsonl"):
        d = json.loads(line)
        m = d.get("meta") or {}
        if m.get("polarity") is None or not d.get("text"):
            continue
        rows.append(d)
    random.Random(seed).shuffle(rows)
    pols = sorted({str(r["meta"]["polarity"]) for r in rows})
    doms = sorted({r["meta"].get("domain") for r in rows if r["meta"].get("domain")})
    out = []
    for r in rows[:cap]:
        m = r["meta"]
        state = r["text"][:1500]
        pol = str(m["polarity"])
        out.append(_choice("aihub147_sentiment", "ko", state,
                           "다음 리뷰 글이 해당 상품에 대해 보이는 감성을 판정하세요.",
                           {p: None for p in pols}, pols.index(pol)))
        if m.get("domain") and len(out) < cap:
            out.append(_choice("aihub147_domain", "ko", state,
                               "다음 글이 다루는 상품 분야를 고르세요.",
                               {d: None for d in doms}, doms.index(m["domain"])))
    return out[:cap]


# ---------------------------------------------------------------- AI-Hub meta-label topic tasks
_META_TASKS = [
    ("aihub-142-news-knowledge.jsonl", "field", "다음 뉴스 본문의 분야를 고르세요.", "aihub142_field", 6000),
    ("aihub-058-agri-parallel.jsonl", "대분류", "다음 문장이 속한 산업 대분류를 고르세요.", "aihub058_major", 4000),
    ("aihub-158-time-expression.jsonl", "category", "다음 기사 본문의 분야를 고르세요.", "aihub158_cat", 3000),
    ("aihub-015-fairy-tale.jsonl", "classification", "다음 동화 본문의 분류를 고르세요.", "aihub015_cls", 2000),
    ("aihub-026-essay-argument.jsonl", "subject", "다음 글의 주제 영역을 고르세요.", "aihub026_subj", 3000),
]


def aihub_meta(seed=17, max_labels=40):
    out = []
    for fn, key, ins, family, cap in _META_TASKS:
        path = f"{KD}/{fn}"
        if not os.path.exists(path):
            continue
        rows = []
        for line in open(path):
            d = json.loads(line)
            v = (d.get("meta") or {}).get(key)
            if v is None or not d.get("text") or len(d["text"]) < 30:
                continue
            rows.append((str(v), d["text"][:1500]))
        from collections import Counter
        cnt = Counter(v for v, _ in rows)
        labels = [l for l, c in cnt.most_common(max_labels) if c >= 20]
        if len(labels) < 2:
            continue
        rows = [(v, t) for v, t in rows if v in labels]
        random.Random(seed).shuffle(rows)
        crit = {l: None for l in labels}
        for v, t in rows[:cap]:
            out.append(_choice(family, "ko", t, ins, dict(crit), labels.index(v)))
    return out


# ---------------------------------------------------------------- EN replay (in-distribution)
def en_replay():
    out = []
    for it in B.typed_decisions(split="train"):
        out.append({"family": "en_typed_decisions", "lang": "en", "state": it["state"],
                    "q": it["q"], "gold_idx": it["gold_idx"]})
    return out


def build_all(seed=17):
    parts = {"klue": klue_train(), "aihub066": aihub_066(), "aihub147": aihub_147(),
             "aihub_meta": aihub_meta(), "en_replay": en_replay()}
    items = [x for v in parts.values() for x in v]
    random.Random(seed).shuffle(items)
    return items, parts


if __name__ == "__main__":
    from collections import Counter
    items, parts = build_all()
    print("total:", len(items))
    print("\nby family:")
    for f, c in Counter(i["family"] for i in items).most_common():
        ks = {len(i["q"]["crit"]) for i in items if i["family"] == f}
        ts = {i["q"]["t"] for i in items if i["family"] == f}
        print(f"  {f:24s} {c:6d}  type={sorted(ts)} k={sorted(ks)[:4]}")
    print("\nby lang:", dict(Counter(i["lang"] for i in items)))
    print("by type:", dict(Counter(i["q"]["t"] for i in items)))
    out = paths.ensure(paths.data("train_items.jsonl"))
    with open(out, "w") as f:
        for i in items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")

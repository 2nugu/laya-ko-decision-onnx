"""Build typed-decision eval items from local benchmark files.

Each item: {"bench", "state", "q": {"t","ins","crit"}, "gold_idx"}
 - choice -> crit is an ordered dict {key: text}; gold_idx indexes into it
 - noul   -> crit {"false":..,"true":..}; gold_idx in {0,1}
"""
import json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

EVAL_DIR = paths.EVAL_DIR
KO_CULTURE = os.path.join(paths.AIHUB_DIR, "aihub-066-global-culture.jsonl")

INS_MC = "다음 질문에 대한 가장 적절한 답을 선택지에서 고르세요."
INS_MC_EN = "Select the single best answer to the question from the options."
INS_MC_JA = "次の質問に対する最も適切な答えを選択肢から選んでください。"
INS_BOOL = "다음 진술이 참인지 거짓인지 판정하세요."


_WARNED = set()


def _missing(path, bench):
    """Benchmarks backed by data this repo does not redistribute degrade to empty."""
    if os.path.exists(path):
        return False
    if bench not in _WARNED:
        _WARNED.add(bench)
        print(f"[skip] {bench}: {path} not found "
              f"(set LAYA_KO_EVAL_DIR / LAYA_KO_AIHUB_DIR to point at it)")
    return True


def _choice(bench, state, ins, options, gold_idx):
    keys = [chr(ord("A") + i) for i in range(len(options))]
    return {"bench": bench, "state": state, "gold_idx": gold_idx,
            "q": {"t": "choice", "ins": ins, "crit": {k: str(o) for k, o in zip(keys, options)}}}


# ------------------------------------------------------------------ KO: KMMLU (45 subsets, native Korean)
def kmmlu(per_subject=20, seed=0):
    if _missing(f"{EVAL_DIR}/kmmlu", "ko_kmmlu"):
        return []
    items = []
    for fn in sorted(os.listdir(f"{EVAL_DIR}/kmmlu")):
        rows = [json.loads(l) for l in open(f"{EVAL_DIR}/kmmlu/{fn}")]
        random.Random(seed).shuffle(rows)
        for r in rows[:per_subject]:
            opts = [r["A"], r["B"], r["C"], r["D"]]
            items.append(_choice("ko_kmmlu", r["question"], INS_MC, opts, int(r["answer"]) - 1))
    return items


# ------------------------------------------------------------------ KO: AI-Hub 066 global culture
_RE_Q = re.compile(r"\[질문\]\s*(.*?)(?=\n\[선택지\]|\n\[정답\]|$)", re.S)
_RE_C = re.compile(r"\[개념\]\s*(.*?)\n", re.S)
_RE_OPT = re.compile(r"^\s*([A-Z])\.\s*(.*)$")
_RE_A = re.compile(r"\[정답\]\s*(.*?)\s*$", re.S)


def ko_culture(per_task=150, seed=0):
    if _missing(KO_CULTURE, "ko_culture"):
        return []
    by_task = {}
    for line in open(KO_CULTURE):
        d = json.loads(line)
        by_task.setdefault(d["meta"].get("task"), []).append(d)
    items = []
    for task in ("Vie_CSQA", "Vie_MMLU", "Vie_CMMU", "Vie_Winogrande", "Vie_HHH", "Vie_BoolQ"):
        rows = by_task.get(task, [])
        random.Random(seed).shuffle(rows)
        n = 0
        for d in rows:
            if n >= per_task:
                break
            txt = d["text"]
            mq = _RE_Q.search(txt)
            if not mq:
                continue
            q = mq.group(1).strip()
            mc = _RE_C.search(txt)
            state = (f"[개념] {mc.group(1).strip()}\n" if mc else "") + q
            if task == "Vie_BoolQ":
                ma = _RE_A.search(txt)
                if not ma:
                    continue
                v = ma.group(1).strip().lower()
                if v not in ("true", "false"):
                    continue
                items.append({"bench": "ko_culture", "state": state, "gold_idx": int(v == "true"),
                              "q": {"t": "noul", "ins": INS_BOOL,
                                    "crit": {"false": "아니다, 진술이 성립하지 않는다",
                                             "true": "그렇다, 진술이 성립한다"}}})
                n += 1
                continue
            block = txt.split("[선택지]")
            if len(block) < 2:
                continue
            opts = []
            for line2 in block[1].split("[정답]")[0].splitlines():
                m = _RE_OPT.match(line2)
                if m:
                    opts.append(m.group(2).strip())
            gi = d["meta"].get("answer_index")
            if gi is None:  # HHH stores a one-hot list in [정답]
                ma = _RE_A.search(txt)
                try:
                    onehot = json.loads(ma.group(1).strip().replace("'", '"'))
                    gi = int(onehot.index(max(onehot)))
                except Exception:
                    continue
            if not opts or not (0 <= int(gi) < len(opts)):
                continue
            items.append(_choice("ko_culture", state, INS_MC, opts, int(gi)))
            n += 1
    return items


# ------------------------------------------------------------------ EN: MMLU (retention, zero-shot)
def mmlu(per_subject=40, seed=0):
    if _missing(f"{EVAL_DIR}/mmlu", "en_mmlu"):
        return []
    items = []
    for fn in sorted(os.listdir(f"{EVAL_DIR}/mmlu")):
        rows = [json.loads(l) for l in open(f"{EVAL_DIR}/mmlu/{fn}")]
        random.Random(seed).shuffle(rows)
        for r in rows[:per_subject]:
            items.append(_choice("en_mmlu", r["question"], INS_MC_EN, r["choices"], int(r["answer"])))
    return items


# ------------------------------------------------------------------ JA: JCommonsenseQA (retention, zero-shot)
def jcqa(n=500, seed=0):
    if _missing(f"{EVAL_DIR}/jcommonsenseqa/validation.jsonl", "ja_jcqa"):
        return []
    rows = [json.loads(l) for l in open(f"{EVAL_DIR}/jcommonsenseqa/validation.jsonl")]
    random.Random(seed).shuffle(rows)
    items = []
    for r in rows[:n]:
        opts = [r[f"choice{i}"] for i in range(5)]
        items.append(_choice("ja_jcqa", r["question"], INS_MC_JA, opts, int(r["label"])))
    return items


# ------------------------------------------------------------------ EN: LocalLLaMA/typed-decisions (in-domain retention)
def typed_decisions(split="test", cache=None):
    import pyarrow.parquet as pq
    cache = cache or paths.data("typed_decisions")
    path = os.path.join(cache, f"{split}.parquet")
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        os.makedirs(cache, exist_ok=True)
        src = hf_hub_download("LocalLLaMA/typed-decisions", f"all/{split}-00000-of-00001.parquet",
                              repo_type="dataset")
        import shutil
        shutil.copy(src, path)
    tbl = pq.read_table(path).to_pylist()
    items = []
    for row in tbl:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, q in questions.items():
            g = gold.get(qid)
            if g is None:
                continue
            t = q["type"]
            crit = q.get("criteria", {})
            if t == "choice":
                keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
                lbl = str(g.get("label"))
                if lbl not in keys:
                    continue
                gi = keys.index(lbl)
                crit_d = crit if isinstance(crit, dict) else {k: None for k in keys}
            elif t == "noul":
                gi = int(str(g.get("label")).lower() == "true")
                crit_d = crit or {}
            elif t == "score":
                probs = g.get("probabilities", {})
                if not probs:
                    continue
                gi = int(max(probs, key=lambda k: probs[k]))
                crit_d = crit
            else:
                continue
            ins = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"])
            items.append({"bench": "en_typed_decisions", "state": state, "gold_idx": gi,
                          "q": {"t": t, "ins": ins, "crit": crit_d}})
    return items


BUILDERS = {"ko_kmmlu": kmmlu, "ko_culture": ko_culture, "en_mmlu": mmlu,
            "ja_jcqa": jcqa, "en_typed_decisions": typed_decisions}

if __name__ == "__main__":
    for name, fn in BUILDERS.items():
        try:
            it = fn()
            from collections import Counter
            print(f"{name:22s} n={len(it):5d}  types={dict(Counter(i['q']['t'] for i in it))}")
        except Exception as e:
            print(f"{name:22s} FAILED: {type(e).__name__}: {e}")


# ============================================================ KLUE (CC-BY-SA-4.0) — the right KO suite
_YNAT_DESC = {
    "IT과학": "정보기술·과학 기사", "경제": "경제·금융·산업 기사", "사회": "사회·사건사고 기사",
    "생활문화": "생활·문화·연예 기사", "세계": "국제·해외 기사", "스포츠": "스포츠 기사",
    "정치": "정치·외교 기사",
}
_NLI_DESC = {
    "entailment": "가설이 전제로부터 반드시 참이다 (함의)",
    "neutral": "가설이 전제로부터 참인지 거짓인지 알 수 없다 (중립)",
    "contradiction": "가설이 전제와 모순된다 (모순)",
}
_STS_LEVELS = ["전혀 관련 없다", "주제만 약간 겹친다", "부분적으로 유사하다",
               "대체로 같은 내용이다", "거의 같은 내용이다", "의미가 완전히 같다"]


def _klue(cfg, split):
    from datasets import load_dataset
    return load_dataset("klue/klue", cfg)[split]


def klue_ynat(n=1000, split="validation", seed=0):
    ds = _klue("ynat", split)
    names = ds.features["label"].names
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        r = ds[i]
        items.append({"bench": "klue_ynat", "state": r["title"], "gold_idx": int(r["label"]),
                      "q": {"t": "choice", "ins": "다음 뉴스 제목의 주제 분야를 고르세요.",
                            "crit": {k: _YNAT_DESC[k] for k in names}}})
    return items


def klue_nli(n=1000, split="validation", seed=0):
    ds = _klue("nli", split)
    names = ds.features["label"].names
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        r = ds[i]
        state = f"전제: {r['premise']}\n가설: {r['hypothesis']}"
        items.append({"bench": "klue_nli", "state": state, "gold_idx": int(r["label"]),
                      "q": {"t": "choice", "ins": "전제에 대해 가설이 갖는 논리적 관계를 판정하세요.",
                            "crit": {k: _NLI_DESC[k] for k in names}}})
    return items


def klue_sts(n=519, split="validation", seed=0):
    ds = _klue("sts", split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        r = ds[i]
        lvl = int(round(float(r["labels"]["label"])))
        lvl = max(0, min(5, lvl))
        state = f"문장1: {r['sentence1']}\n문장2: {r['sentence2']}"
        items.append({"bench": "klue_sts", "state": state, "gold_idx": lvl,
                      "q": {"t": "score", "ins": "두 문장의 의미적 유사도를 0~5 단계로 판정하세요.",
                            "crit": list(_STS_LEVELS)}})
    return items


def klue_re(n=1000, split="validation", seed=0):
    ds = _klue("re", split)
    names = ds.features["label"].names
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        r = ds[i]
        state = (f"문장: {r['sentence']}\n"
                 f"주어 개체: {r['subject_entity']['word']} ({r['subject_entity']['type']})\n"
                 f"목적 개체: {r['object_entity']['word']} ({r['object_entity']['type']})")
        items.append({"bench": "klue_re", "state": state, "gold_idx": int(r["label"]),
                      "q": {"t": "choice", "ins": "두 개체 사이의 관계를 고르세요.",
                            "crit": {k: None for k in names}}})
    return items


BUILDERS.update({"klue_ynat": klue_ynat, "klue_nli": klue_nli,
                 "klue_sts": klue_sts, "klue_re": klue_re})


# ============================================================ Kev decision-v2 (Apache-2.0) — EN retention
_KEV_URL = "https://raw.githubusercontent.com/jaredpalmer/kev/main/evals/decision-v2/{split}.jsonl"
_KEV_CACHE = paths.data("kev")


def kev_v2(split="test", n=None, seed=0):
    os.makedirs(_KEV_CACHE, exist_ok=True)
    path = os.path.join(_KEV_CACHE, f"decision-v2-{split}.jsonl")
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(_KEV_URL.format(split=split), path)
    rows = [json.loads(l) for l in open(path)]
    items = []
    for r in rows:
        for qid, q in r["questions"].items():
            t, crit, lbl = q["type"], q.get("criteria"), q.get("label")
            if lbl is None:
                continue
            ins = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"], ensure_ascii=False)
            if t == "choice":
                keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
                if str(lbl) not in keys:
                    continue
                items.append({"bench": "en_kev_v2", "state": r["state"], "gold_idx": keys.index(str(lbl)),
                              "q": {"t": "choice", "ins": ins,
                                    "crit": crit if isinstance(crit, dict) else {k: None for k in keys}}})
            elif t == "noul":
                items.append({"bench": "en_kev_v2", "state": r["state"],
                              "gold_idx": int(bool(lbl) if isinstance(lbl, bool) else str(lbl).lower() == "true"),
                              "q": {"t": "noul", "ins": ins, "crit": crit or {}}})
            elif t == "score":
                if not isinstance(crit, list) or not (0 <= int(lbl) < len(crit)):
                    continue
                items.append({"bench": "en_kev_v2", "state": r["state"], "gold_idx": int(lbl),
                              "q": {"t": "score", "ins": ins, "crit": crit}})
    if n:
        random.Random(seed).shuffle(items)
        items = items[:n]
    return items


BUILDERS["en_kev_v2"] = kev_v2

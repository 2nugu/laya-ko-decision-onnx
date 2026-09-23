"""Render a before/after table from saved bench json / training history."""
import argparse, json, os

KO = ("klue_ynat", "klue_nli", "klue_sts", "klue_re", "ko_culture", "ko_kmmlu")
KEEP = ("ja_jcqa", "en_typed_decisions", "en_kev_v2", "en_mmlu")


def load(path):
    d = json.load(open(path))
    return d["results"] if "results" in d else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs="+", required=True)
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()

    before, after = {}, {}
    for p in a.before:
        before.update(load(p))
    for p in a.after:
        if p.endswith("history.json"):
            h = json.load(open(p))
            after.update(h[-1]["results"])
        else:
            after.update(load(p))

    rows = []
    for b in list(KO) + list(KEEP):
        if b not in before and b not in after:
            continue
        x, y = before.get(b), after.get(b)
        rows.append((b, x, y))

    hdr = f"| {'bench':22s} | {'n':>5s} | {'chance':>6s} | {'before':>6s} | {'after':>6s} | {'Δacc':>7s} | {'ECE before':>10s} | {'ECE after':>9s} |"
    sep = "|" + "|".join("-" * (len(c) + 2) for c in hdr.split("|")[1:-1]) + "|"
    lines = [hdr, sep]
    for b, x, y in rows:
        n = (y or x)["n"]
        ch = (y or x)["chance"]
        ba = f"{x['acc']:.3f}" if x else "  -  "
        aa = f"{y['acc']:.3f}" if y else "  -  "
        d = f"{(y['acc']-x['acc'])*100:+.1f}pp" if (x and y) else "   -   "
        be = f"{x['ece']:.3f}" if x else "  -  "
        ae = f"{y['ece']:.3f}" if y else "  -  "
        tag = "" if b in KO else "  (retention)"
        lines.append(f"| {b+tag:22s} | {n:5d} | {ch:6.3f} | {ba:>6s} | {aa:>6s} | {d:>7s} | {be:>10s} | {ae:>9s} |")
    out = "\n".join(lines)
    print(out)
    if a.md:
        os.makedirs(os.path.dirname(a.md), exist_ok=True)
        open(a.md, "w").write(out + "\n")
        print(f"\nwrote {a.md}")


if __name__ == "__main__":
    main()

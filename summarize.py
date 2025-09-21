#!/usr/bin/env python3
"""
results numbers for tables, auc, cohens d, p-values
+ the others

python3 src/organism/data/summarise_paraphr_from_masters.py \
    --base output/qwen06b_runs/paraphrasability_base_master.jsonl \
    --lora output/qwen06b_runs/paraphrasability_lora_master.jsonl \
    --write_tables \
    --out_dir output/qwen06b_runs/summary_from_masters
"""

import argparse, json, math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Stats

def rankdata_ties(x: List[float]) -> List[float]:
    xs = sorted((v, i) for i, v in enumerate(x))
    ranks = [0.0] * len(x)
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j][0] == xs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[xs[k][1]] = avg
        i = j
    return ranks

def mannwhitney_u(a: List[float], b: List[float]) -> Tuple[float, float]:
    n1, n2 = len(a), len(b)
    data = [(v, 0) for v in a] + [(v, 1) for v in b]
    ranks = rankdata_ties([v for v, _ in data])
    R1 = sum(r for (r, (_, g)) in zip(ranks, data) if g == 0)
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    return U1, U2

def mw_pvalue_normal(a: List[float], b: List[float]) -> float:
    # two-sided normal approximation with tie correction
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    U, _ = mannwhitney_u(a, b)
    allv = a + b
    from collections import Counter
    counts = Counter(allv)
    N = n1 + n2
    mu = n1 * n2 / 2.0
    # variance with tie correction
    tie_term = sum(c * (c*c - 1) for c in counts.values())
    sigma2 = n1 * n2 * (N + 1) / 12.0
    if N > 1:
        sigma2 -= (n1 * n2 * tie_term) / (12.0 * N * (N - 1))
    sigma = math.sqrt(max(sigma2, 1e-12))
    z = (U - mu) / sigma
    # two-sided p
    p_one = 0.5 * (1.0 + math.erf(-abs(z) / math.sqrt(2)))
    return 2.0 * p_one

def cohens_d(a: List[float], b: List[float]) -> float:
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    m1 = sum(a) / n1
    m2 = sum(b) / n2
    def var(x, m): return sum((v - m)**2 for v in x) / (len(x) - 1) if len(x) > 1 else 0.0
    v1, v2 = var(a, m1), var(b, m2)
    dof = (n1 - 1) + (n2 - 1)
    sp = math.sqrt(((n1 - 1)*v1 + (n2 - 1)*v2) / dof) if dof > 0 else 0.0
    return (m1 - m2) / (sp if sp > 0 else 1e-12)

def mean_std(x: List[float]) -> Tuple[float, float]:
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(x) / n
    if n == 1:
        return m, 0.0
    v = sum((v - m)**2 for v in x) / (n - 1)
    return m, math.sqrt(max(v, 0.0))

def ks_stat(a: List[float], b: List[float]) -> float:
    # Two-sample KS statistic
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    i = j = 0
    n1, n2 = len(a_sorted), len(b_sorted)
    d = 0.0
    while i < n1 and j < n2:
        if a_sorted[i] <= b_sorted[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / n1 - j / n2))
    # ensure full sweep
    d = max(d, abs(1.0 - j / n2), abs(i / n1 - 1.0))
    return d

def ks_pvalue_asymp(D: float, n1: int, n2: int) -> float:
    # Asymptotic Smirnov p-value with Massey correction
    if n1 == 0 or n2 == 0:
        return float("nan")
    ne = n1 * n2 / (n1 + n2)
    if ne <= 0:
        return float("nan")
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * D
    # Q_KS(lam) = 2 * sum_{j=1..inf} (-1)^{j-1} exp(-2 j^2 lam^2)
    s = 0.0
    for j in range(1, 100):
        s += ((-1)**(j - 1)) * math.exp(-2.0 * (j*j) * (lam*lam))
    p = max(0.0, min(1.0, 2.0 * s))
    return p

def summarize_two_samples(a: List[float], b: List[float]) -> Dict[str, float]:
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {"n_a": n1, "n_b": n2, "auc": float("nan"), "p": float("nan"),
                "cohens_d": float("nan"), "ks": float("nan"), "ks_p": float("nan")}
    U, _ = mannwhitney_u(a, b)
    auc = U / (n1 * n2)
    p_mw = mw_pvalue_normal(a, b)
    d = cohens_d(a, b)
    D = ks_stat(a, b)
    p_ks = ks_pvalue_asymp(D, n1, n2)
    return {"n_a": n1, "n_b": n2, "auc": auc, "p": p_mw, "cohens_d": d, "ks": D, "ks_p": p_ks}

# IO helpers

def load_deltas_paraphr(path: Path) -> List[float]:
    vals: List[float] = []
    if not path.exists():
        return vals
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj.get("delta"), (int, float)):
                    vals.append(float(obj["delta"]))
            except Exception:
                pass
    return vals

def load_deltas_pp(path: Path, style: Optional[str], target: Optional[str]) -> List[float]:
    vals: List[float] = []
    if not path.exists():
        return vals
    want_style = style.lower() if style else None
    want_target = target.lower() if target else None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj.get("delta"), (int, float)):
                continue
            if want_style is not None:
                s = str(obj.get("style", "")).lower()
                if s != want_style:
                    continue
            if want_target is not None:
                t = str(obj.get("logprob_target", "")).lower()
                if t != want_target:
                    continue
            vals.append(float(obj["delta"]))
    return vals

def write_row(path: Path, header: str, row: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8") as f:
        if need_header:
            f.write(header + "\n")
        f.write(row + "\n")

def take_n(x: List[float], n: int) -> List[float]:
    if n is None or n <= 0:
        return x
    return x[: min(n, len(x))]

def print_group_stats(label: str, x: List[float]):
    m, s = mean_std(x)
    print(f"    {label:<6}  n={len(x):<4}  mean={m:+.4f}  std={s:.4f}")

# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Paraphrasability BASE master JSONL")
    ap.add_argument("--lora", required=True, help="Paraphrasability LORA master JSONL")
    ap.add_argument("--n", type=int, default=100, help="Max samples per group per comparison (first N rows)")
    ap.add_argument("--out_dir", default=".", help="Base directory for tables (used only with --write_tables)")
    ap.add_argument("--tag", default="", help="Optional tag to put in the 'run_dir' column of tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tag = args.tag or "masters"

    # Paraphrasability
    a_full = load_deltas_paraphr(Path(args.base))
    b_full = load_deltas_paraphr(Path(args.lora))
    a = take_n(a_full, args.n)
    b = take_n(b_full, args.n)

    print("\nParaphrasability (BASE vs LoRA)")
    print_group_stats("BASE", a)
    print_group_stats("LoRA", b)
    stats = summarize_two_samples(a, b)
    print(f"  AUC={stats['auc']:.3f}  p_MWU={stats['p']:.2e}   d={stats['cohens_d']:+.3f}   KS={stats['ks']:.3f}  p_KS={stats['ks_p']:.2e}")
    if (stats["n_a"] < 10 or stats["n_b"] < 10) and not (math.isnan(stats["auc"]) or math.isnan(stats["p"])):
        print("  [warn] very small sample size; treat p/d with caution.")

    m, s = mean_std(b)
    print(f"Lora     ${m:+.3f}$ $\\pm$ ${s:.3f}$")
    print(f"Lora ks  ${stats['ks']:+.3f}$ $\\pm$ ${stats['ks_p']:.3f}$")
    print(f"Lora d   ${stats['cohens_d']:+.3f}$")
    print(f"Lora auc ${stats['auc']:+.3f}$ $\\pm$ ${stats['p']}$")

    print(f"AUC={stats['auc']:.3f}  p_MWU={stats['p']:.2e}   d={stats['cohens_d']:+.3f}   KS={stats['ks']:.3f}  p_KS={stats['ks_p']:.2e}")

if __name__ == "__main__":
    main()

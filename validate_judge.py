#!/usr/bin/env python3
"""
validate_judge.py
==================================================================
Computes the numbers for Section 5.4 of the chapter:
  * Cohen's kappa between the two human annotators
  * agreement of the automatic scores (P1 scaffolding, P2 accuracy) with the humans

HOW TO USE
  1. Run the experiment. It writes  experiment_results_human_validation_sample.csv
  2. Make two copies, one per annotator. Each annotator fills, for every row:
        human_scaffolding_0_1   1 = the visible tutor text does NOT give the full answer/fix, 0 = it does
        human_accurate_0_1      1 = every claim is consistent with the code, 0 = something is wrong
        annotator               a name or letter (A / B)
     Annotators must work independently and must not see the automatic scores.
  3. python validate_judge.py experiment_results.csv annotatorA.csv annotatorB.csv

Where the two humans disagree, the row is listed so you can resolve it by discussion;
the "consensus" is then the agreed label. Without a discussion step, the script uses
only the rows on which both humans agree (and says how many that is).
==================================================================
"""
import argparse
import sys

import numpy as np
import pandas as pd

LABELS = {"scaffolding": ("human_scaffolding_0_1", "P1_scaffolding"),
          "accuracy": ("human_accurate_0_1", "P2_accuracy")}


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n == 0:
        return np.nan, np.nan
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in np.union1d(a, b))
    k = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    return k, po


def boot_kappa(a, b, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    ks = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        k, _ = kappa(a[i], b[i])
        if not np.isnan(k):
            ks.append(k)
    return tuple(np.percentile(ks, [2.5, 97.5])) if ks else (np.nan, np.nan)


def load(path):
    d = pd.read_csv(path)
    for c, _ in LABELS.values():
        if c not in d:
            sys.exit(f"{path}: column {c} is missing")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.set_index("run_id")


def interpret(k):
    if np.isnan(k):
        return "n/a"
    for lim, name in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k <= lim:
            return name
    return "almost perfect"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_csv")
    ap.add_argument("annotator_a")
    ap.add_argument("annotator_b")
    a = ap.parse_args()

    res = pd.read_csv(a.results_csv)
    # a resumed or retried run can leave an unscored row with the same run_id: keep the scored one
    res["_scored"] = res["P"].notna() if "P" in res else True
    res = res.sort_values("_scored", kind="stable").drop_duplicates("run_id", keep="last").set_index("run_id")
    A, B = load(a.annotator_a), load(a.annotator_b)
    ids = A.index.intersection(B.index)
    print(f"Rows in A: {len(A)}, in B: {len(B)}, shared: {len(ids)}")
    if not len(ids):
        sys.exit("The two files share no run_id.")
    rows = []
    for name, (hcol, mcol) in LABELS.items():
        x = pd.DataFrame({"a": A.loc[ids, hcol], "b": B.loc[ids, hcol]}).dropna()
        if x.empty:
            print(f"\n[{name}] no completed labels from both annotators yet.")
            continue
        k, po = kappa(x.a, x.b)
        lo, hi = boot_kappa(x.a, x.b)
        print(f"\n=== {name} ===")
        print(f"Inter-annotator: n={len(x)}  agreement={po:.1%}  Cohen's kappa={k:.2f} "
              f"[95% bootstrap CI {lo:.2f}, {hi:.2f}] ({interpret(k)})")
        dis = x[x.a != x.b]
        if len(dis):
            print(f"  {len(dis)} disagreements; resolve these by discussion: run_id = {list(dis.index)}")
        cons = x[x.a == x.b].a.astype(int)
        if mcol in res:
            m = res.loc[res.index.intersection(cons.index), mcol].dropna().astype(int)
            c = cons.loc[m.index]
            if len(c):
                kj, pj = kappa(c, m)
                tp = ((m == 1) & (c == 1)).sum(); tn = ((m == 0) & (c == 0)).sum()
                fp = ((m == 1) & (c == 0)).sum(); fn = ((m == 0) & (c == 1)).sum()
                print(f"Automatic score ({mcol}) vs human consensus: n={len(c)} (agreed rows only)  "
                      f"agreement={pj:.1%}  kappa={kj:.2f} ({interpret(kj)})")
                print(f"  automatic=1/human=1: {tp}   0/0: {tn}   "
                      f"automatic=1/human=0 (too lenient): {fp}   automatic=0/human=1 (too strict): {fn}")
                rows.append({"score": name, "n_pairs": len(x), "kappa_human_human": round(k, 3),
                             "agreement_human_human": round(po, 3), "n_consensus": len(c),
                             "kappa_auto_vs_human": round(kj, 3), "agreement_auto_vs_human": round(pj, 3)})
            else:
                print("No consensus rows overlap with the automatic scores.")
    if rows:
        pd.DataFrame(rows).to_csv("table_validation.csv", index=False)
        print("\nSaved: table_validation.csv")
    print("\nReport kappa with n and say that the sample is small; with fewer than about 30 items "
          "the intervals are wide.")


if __name__ == "__main__":
    main()

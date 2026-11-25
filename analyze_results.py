#!/usr/bin/env python3
"""
analyze_results.py
==================================================================
Turns experiment_results.csv into the tables for the chapter.

DEFINITIONS (state these verbatim in the paper)
-----------------------------------------------
P     pedagogical quality in [0, 1] = mean of three per-run components
        P1 scaffolding   : 1 if the student-visible text does NOT give the full answer
                           (no regex leak, no code block on debugging questions, judge
                           says it did not give the full answer), else 0
        P2 accuracy      : 1 if the judge finds every claim consistent with the ground truth
        P3 student success: 1 if the simulated student then passes the unit tests
                           (debugging) or states the required facts (other queries)
E     engineering cost  = alpha * (L / L_ref) + (1 - alpha) * (T / T_ref)
        L = tutor-call latency in seconds, T = total tokens of the tutor call
        (input + output + reasoning), L_ref = 1 s, T_ref = 1000 tokens, alpha = 0.5
ITEI  = P / E, computed on configuration means (as in the chapter)

DATA CLEANING (applied automatically, counts are printed)
  * truncated rows (output hit the token cap) are excluded           (--keep-truncated to keep)
  * latency is averaged only over calls with n_retries == 0 and latency <= --latency-cap
    seconds, because retried / throttled free-tier calls measure the provider queue,
    not the pipeline. Token counts use all non-truncated rows.
  * rows with provider == 'control' are the no-tutor control for the simulated student.
    P3 (student success) is only informative for queries where the control success rate is low;
    P_no_P3 = mean(P1, P2) is reported next to P so readers can see the effect.

These definitions are NEW (v2) and will not reproduce the E and ITEI values of the
earlier draft; replace the old Table 2 with the output of this script.

USAGE
-----
    python analyze_results.py experiment_results.csv
    python analyze_results.py experiment_results.csv --alpha 0.5 --lref 1 --tref 1000
    python analyze_results.py experiment_results_rejudged.csv --providers groq cerebras   # chapter's pooled numbers
    python analyze_results.py experiment_results_rejudged.csv --providers gemini          # partial Gemini rows
    python analyze_results.py mock_results.csv --allow-mock      # pipeline test only
==================================================================
"""
import argparse
import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)


def boot_ci(x, n=5000, level=0.95):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    means = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return (lo, hi)


def add_E(df, alpha, lref, tref):
    df = df.copy()
    df["E"] = alpha * df["latency_seconds"] / lref + (1 - alpha) * df["total_tokens"] / tref
    return df


def config_table(df, alpha, lref, tref):
    keys = ["provider", "model", "chunking_strategy", "prompt_strategy"]
    g = df.groupby(keys)
    t = g.agg(n=("run_id", "count"),
              P=("P", "mean"), P_no_P3=("P_no_P3", "mean"), P1=("P1_scaffolding", "mean"), P2=("P2_accuracy", "mean"),
              P3=("P3_student_success", "mean"),
              n_lat=("latency_seconds", "count"), L=("latency_seconds", "mean"), T=("total_tokens", "mean"),
              target_intact=("target_intact", "mean"), context_chars=("context_chars", "mean")).reset_index()
    t["P_ci_lo"], t["P_ci_hi"] = zip(*[boot_ci(grp["P"]) for _, grp in g])
    t["L_ci_lo"], t["L_ci_hi"] = zip(*[boot_ci(grp["latency_seconds"]) for _, grp in g])
    t["E"] = alpha * t["L"] / lref + (1 - alpha) * t["T"] / tref
    t["ITEI"] = t["P"] / t["E"]
    return t


def sensitivity(df, lref, tref):
    rows = []
    for alpha in (0.25, 0.5, 0.75):
        t = config_table(df, alpha, lref, tref)
        for floor in (None, 0.85):
            u = t if floor is None else t[t["P"] >= floor]
            if u.empty:
                continue
            best = u.sort_values("ITEI", ascending=False).iloc[0]
            rows.append({"alpha": alpha, "quality_floor": floor or 0,
                         "best_config": f'{best["provider"]}/{best["chunking_strategy"]}/{best["prompt_strategy"]}',
                         "ITEI": round(best["ITEI"], 3), "P": round(best["P"], 3)})
    return pd.DataFrame(rows)


def paired_effect(df, factor, a, b, metric):
    """Mean difference (a - b) with bootstrap CI, paired on the other design factors."""
    keys = ["query_id", "provider", "repeat"] + [c for c in ("chunking_strategy", "prompt_strategy")
                                                if c != factor]
    pv = df.pivot_table(index=keys, columns=factor, values=metric, aggfunc="mean")
    if a not in pv or b not in pv:
        return None
    d = (pv[a] - pv[b]).dropna()
    lo, hi = boot_ci(d)
    return {"factor": factor, "contrast": f"{a} - {b}", "metric": metric, "n_pairs": len(d),
            "mean_diff": round(d.mean(), 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lref", type=float, default=1.0)
    ap.add_argument("--tref", type=float, default=1000.0)
    ap.add_argument("--allow-mock", action="store_true")
    ap.add_argument("--keep-truncated", action="store_true")
    ap.add_argument("--latency-cap", type=float, default=60.0)
    ap.add_argument("--providers", nargs="+", default=None,
                    help="analyse only these tutor providers (e.g. groq cerebras); the chapter pools "
                         "Groq and Cerebras and reports Gemini separately")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    if (df["status"] == "MOCK").any() and not a.allow_mock:
        sys.exit("Refusing to analyse MOCK (fake) data. Use --allow-mock only for pipeline tests.")
    n_all = len(df)
    if "truncated" not in df:
        df["truncated"] = 0
    df["truncated"] = df["truncated"].fillna(0).astype(int)
    df = df[df["status"].isin(["OK", "MOCK"])].copy()
    control = df[df["provider"] == "control"].copy()
    df = df[df["provider"] != "control"].copy()
    if a.providers:
        df = df[df["provider"].isin(a.providers)].copy()
        n_all = len(df) + len(control)
    n_trunc = int(df["truncated"].sum())
    if not a.keep_truncated:
        df = df[df["truncated"] == 0].copy()
    lat_bad = (df["n_retries"].fillna(0) > 0) | (df["latency_seconds"] > a.latency_cap)
    print(f"Cleaning: {n_trunc} truncated rows {'kept' if a.keep_truncated else 'excluded'}; "
          f"{int(lat_bad.sum())} of {len(df)} remaining rows have retried/over-cap latency "
          f"(excluded from L and E only).")
    n_noP = int(df["P"].isna().sum())
    if n_noP:
        print(f"WARNING: {n_noP} rows have no P (judge or student step failed) and are excluded.")
        df = df[df["P"].notna()].copy()
        lat_bad = lat_bad.loc[df.index]
    df["latency_raw"] = df["latency_seconds"]
    df.loc[lat_bad, "latency_seconds"] = np.nan
    df["P_no_P3"] = df[["P1_scaffolding", "P2_accuracy"]].mean(axis=1)
    print(f"Rows: {n_all} total = {len(df)} tutor rows analysed + {len(control)} no-tutor control rows + "
          f"{n_all - len(df) - len(control)} excluded (failed calls, truncated or missing P; "
          "failed calls are NOT averaged as zero-latency).")
    print(f"Retriever modes in data: {sorted(df['retriever_mode'].dropna().unique())}")
    print(f"Models: {sorted(df['model'].dropna().unique())}")
    print(f"Intent-router accuracy: {df['intent_correct'].mean():.2%}")
    print(f"CoT format compliance: {df.loc[df.prompt_strategy == 'cot_tracing', 'format_ok'].mean():.2%}\n")

    if len(control):
        c = control.groupby("query_id")["student_success"].agg(["mean", "count"]).round(3)
        c["verdict"] = np.where(c["mean"] >= 0.9, "NON-DISCRIMINATING (student succeeds alone)", "ok")
        print("=== No-tutor control: simulated-student success WITHOUT any tutor message ===")
        print(c.to_string())
        print()
        tut = df.groupby("query_id")["P3_student_success"].mean().round(3)
        print("P3 gain over control per query (tutor P3 - control success):")
        print((tut - control.groupby("query_id")["student_success"].mean()).round(3).to_string())
        print()
    else:
        print("WARNING: no control rows; P3 cannot be separated from the simulator's own knowledge.\n")

    t = config_table(df, a.alpha, a.lref, a.tref)
    show = t.round(3).sort_values(["provider", "chunking_strategy", "prompt_strategy"])
    print("=== Table: all configurations (means; 95% bootstrap CI for P and L) ===")
    print(show.to_string(index=False))
    show.to_csv("table_all_configurations.csv", index=False)

    print("\n=== Paired main effects (mean difference, 95% bootstrap CI) ===")
    eff = []
    for factor, a1, b1 in (("chunking_strategy", "ast_semantic", "fixed_size"),
                           ("prompt_strategy", "socratic", "zero_shot"),
                           ("prompt_strategy", "cot_tracing", "socratic")):
        for metric in ("P", "latency_seconds", "total_tokens", "target_intact"):
            r = paired_effect(df, factor, a1, b1, metric)
            if r:
                eff.append(r)
    eff = pd.DataFrame(eff)
    print(eff.to_string(index=False))
    eff.to_csv("table_main_effects.csv", index=False)

    print("\n=== Pooled by prompting strategy (mean, 95% bootstrap CI for P) ===")
    rows = []
    for pname, grp in df.groupby("prompt_strategy"):
        lo, hi = boot_ci(grp["P"])
        rows.append({"prompt_strategy": pname, "n": len(grp), "P": grp["P"].mean(), "P_ci_lo": lo, "P_ci_hi": hi,
                     "P1": grp["P1_scaffolding"].mean(), "P2": grp["P2_accuracy"].mean(),
                     "P_no_P3": grp["P_no_P3"].mean(), "P1_zero_runs": int((grp["P1_scaffolding"] == 0).sum())})
    byp = pd.DataFrame(rows)
    print(byp.round(3).to_string(index=False))
    byp.to_csv("table_by_prompt.csv", index=False)

    if {"groq", "cerebras"} <= set(df["provider"]):
        print("\n=== Same model, two serving providers: Groq minus Cerebras (paired) ===")
        prow = []
        for metric in ("latency_seconds", "P"):
            pv = df[df["provider"].isin(["groq", "cerebras"])].pivot_table(
                index=["query_id", "chunking_strategy", "prompt_strategy", "repeat"],
                columns="provider", values=metric, aggfunc="mean")
            d = (pv["groq"] - pv["cerebras"]).dropna()
            lo, hi = boot_ci(d)
            prow.append({"metric": metric, "n_pairs": len(d), "mean_diff_groq_minus_cerebras": d.mean(),
                         "ci95_lo": lo, "ci95_hi": hi,
                         "mean_groq": df[df.provider == "groq"][metric].mean(),
                         "mean_cerebras": df[df.provider == "cerebras"][metric].mean()})
        prow = pd.DataFrame(prow)
        print(prow.round(4).to_string(index=False))
        prow.to_csv("table_provider_contrast.csv", index=False)

    print("\n=== ITEI sensitivity (weights of latency vs tokens; optional quality floor) ===")
    sens = sensitivity(df, a.lref, a.tref)
    print(sens.to_string(index=False))
    sens.to_csv("table_itei_sensitivity.csv", index=False)

    print("\nSaved: table_all_configurations.csv, table_main_effects.csv, table_by_prompt.csv, table_itei_sensitivity.csv (and table_provider_contrast.csv for Groq vs Cerebras)")
    print("Reminder: report n per configuration and the CIs; do not claim differences whose "
          "CIs overlap zero.")


if __name__ == "__main__":
    main()

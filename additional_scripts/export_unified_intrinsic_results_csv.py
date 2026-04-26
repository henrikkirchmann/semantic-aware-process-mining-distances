"""Export a single tidy CSV with one row per (log, method) and the four
intrinsic metrics, aggregated from the per-log pkls under
``results/activity_distances/intrinsic_df_avg/<log>/dfavg_r<r>_w<w>_samplesize_<s>.pkl``.

Default protocol is the canonical paper setting r=10, w=5, samplesize=5.

Output:
    results/activity_distances/intrinsic_summary/intrinsic_results_per_log_and_method.csv

Columns:
    log_name, method, I_comp, I_nn, I_prec, I_tri

Notes:
    * I_comp = 1 - diameter (paper convention; higher is better).
    * precision@w-1 -> I_nn, precision@1 -> I_prec, triplet -> I_tri.
"""

from __future__ import annotations

import argparse
import os
import pickle

import pandas as pd

from definitions import ROOT_DIR


def export(r: int = 10, w: int = 5, samplesize: int = 5) -> str:
    agg_dir = os.path.join(ROOT_DIR, "results", "activity_distances", "intrinsic_df_avg")
    out_dir = os.path.join(ROOT_DIR, "results", "activity_distances", "intrinsic_summary")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "intrinsic_results_per_log_and_method.csv")

    pkl_name = f"dfavg_r{r}_w{w}_samplesize_{samplesize}.pkl"
    frames = []
    missing = []
    for log_name in sorted(os.listdir(agg_dir)):
        log_dir = os.path.join(agg_dir, log_name)
        if not os.path.isdir(log_dir):
            continue
        pkl_path = os.path.join(log_dir, pkl_name)
        if not os.path.exists(pkl_path):
            missing.append(log_name)
            continue
        with open(pkl_path, "rb") as f:
            df = pickle.load(f)
        df = df.copy()
        df["Log Name"] = log_name
        frames.append(df)

    if not frames:
        raise SystemExit(f"No per-log pkls found under {agg_dir} for {pkl_name}.")

    big = pd.concat(frames, ignore_index=True)
    big["I_comp"] = 1.0 - big["diameter"]
    big = big.rename(
        columns={
            "Distance Function": "method",
            "Log Name": "log_name",
            "precision@w-1": "I_nn",
            "precision@1": "I_prec",
            "triplet": "I_tri",
        }
    )
    big = big[["log_name", "method", "I_comp", "I_nn", "I_prec", "I_tri"]]
    big = big.sort_values(["log_name", "method"]).reset_index(drop=True)
    for c in ("I_comp", "I_nn", "I_prec", "I_tri"):
        big[c] = big[c].round(4)

    big.to_csv(out_path, index=False)
    print(f"wrote {out_path}  rows={len(big)}  logs={big['log_name'].nunique()}  methods={big['method'].nunique()}")
    if missing:
        print(f"  (skipped {len(missing)} log dir(s) without {pkl_name}: {', '.join(missing)})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r", type=int, default=10)
    ap.add_argument("--w", type=int, default=5)
    ap.add_argument("--samplesize", type=int, default=5)
    args = ap.parse_args()
    export(r=args.r, w=args.w, samplesize=args.samplesize)

"""Rebuild the per-log aggregated pickle files in
``results/activity_distances/intrinsic_df_avg/<log>/`` from the raw per-(log,
method, grid) CSV files in ``results/activity_distances/intrinsic/<log>/``.

The downstream analysis script (``analyse_log_stats_vs_intrinsic.py``) reads
these pkl files, but a previous aggregation run silently dropped rows for
several methods on several logs. Most notably, the ``Chiorrini 2022 Embedding
Process Structure`` method is absent from 21 of the 28 per-log pkls even
though the raw CSVs exist for all 28 logs; and on four logs (BPIC12_A,
BPIC12_W_Complete, BPIC18, BPIC19) the pkl contains a single row instead of
the ~47 methods that were actually evaluated.

This script recomputes one pkl per log from scratch with schema identical to
the existing pkls::

    columns: Log Name | Distance Function | diameter | precision@w-1 |
             precision@1 | triplet
    one row per (Log Name, Distance Function)

For each method we take the simple mean of every metric across every row
(``r``, ``w``) found in every raw CSV for that (log, method) pair. Existing
pkls are backed up to a sibling ``*.pkl.bak`` file the first time the script
touches them, so the pre-rebuild data is recoverable.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "activity_distances" / "intrinsic"
AGG_DIR = ROOT / "results" / "activity_distances" / "intrinsic_df_avg"

# Only these 28 logs are part of the intrinsic benchmark; other folders under
# ``intrinsic/`` are artefacts (e.g. aggregate NN folders).
BENCH_LOGS = {
    "BPIC12", "BPIC12_A", "BPIC12_Complete", "BPIC12_O", "BPIC12_W",
    "BPIC12_W_Complete",
    "BPIC13_closed_problems", "BPIC13_incidents", "BPIC13_open_problems",
    "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5",
    "BPIC17", "BPIC18", "BPIC19",
    "BPIC20_DomesticDeclarations", "BPIC20_InternationalDeclarations",
    "BPIC20_PermitLog", "BPIC20_PrepaidTravelCost", "BPIC20_RequestForPayment",
    "CCC19", "Env Permit", "Helpdesk", "Hospital Billing", "RTFM", "Sepsis",
}

# Canonical output filename: matches what the existing pkls are called, so the
# downstream ``load_all`` helper keeps working unchanged.
OUT_FILENAME = "dfavg_r10_w5_samplesize_5.pkl"

METHOD_NAME_RE = re.compile(r"_distfunc_(.+?)_r\d+_w\d+_samplesize_\d+\.csv$")

METRIC_COLS = ["diameter", "precision@w-1", "precision@1", "triplet"]


def extract_method(filename: str) -> str | None:
    m = METHOD_NAME_RE.search(filename)
    return m.group(1) if m else None


def rebuild_log(log_name: str) -> dict[str, int]:
    """Rebuild the aggregated pkl for a single log. Returns a small summary."""
    raw_log_dir = RAW_DIR / log_name
    agg_log_dir = AGG_DIR / log_name

    if not raw_log_dir.is_dir():
        return {"csvs": 0, "methods": 0, "rows_used": 0, "status": "no-raw"}

    agg_log_dir.mkdir(parents=True, exist_ok=True)

    # Group raw CSVs by method
    method_to_csvs: dict[str, list[Path]] = {}
    for fname in sorted(os.listdir(raw_log_dir)):
        if not fname.endswith(".csv"):
            continue
        method = extract_method(fname)
        if method is None:
            continue
        method_to_csvs.setdefault(method, []).append(raw_log_dir / fname)

    if not method_to_csvs:
        return {"csvs": 0, "methods": 0, "rows_used": 0, "status": "no-csv-match"}

    # Aggregate: for each method, take mean of metrics across every row in
    # every raw CSV for that method on this log.
    rows = []
    total_rows_used = 0
    for method, csv_paths in sorted(method_to_csvs.items()):
        frames = []
        for p in csv_paths:
            try:
                frames.append(pd.read_csv(p))
            except Exception as exc:
                print(f"  [warn] {log_name}/{p.name}: {exc}")
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        # Keep only the metric columns and numeric rows
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = pd.NA
        df[METRIC_COLS] = df[METRIC_COLS].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=METRIC_COLS, how="all")
        total_rows_used += len(df)
        means = df[METRIC_COLS].mean(numeric_only=True)
        rows.append({
            "Log Name": log_name,
            "Distance Function": method,
            "diameter":      float(means.get("diameter",      float("nan"))),
            "precision@w-1": float(means.get("precision@w-1", float("nan"))),
            "precision@1":   float(means.get("precision@1",   float("nan"))),
            "triplet":       float(means.get("triplet",       float("nan"))),
        })

    out_df = pd.DataFrame(rows, columns=[
        "Log Name", "Distance Function",
        "diameter", "precision@w-1", "precision@1", "triplet",
    ])

    # Back up every existing pkl the first time we touch it, then remove them
    # so we leave exactly one canonical pkl per log after this script runs.
    existing_pkls = [f for f in os.listdir(agg_log_dir) if f.endswith(".pkl")]
    for f in existing_pkls:
        src = agg_log_dir / f
        bak = src.with_suffix(src.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(src, bak)
        # Remove the original so we don't leave stale/partial pkls alongside.
        src.unlink()

    out_path = agg_log_dir / OUT_FILENAME
    out_df.to_pickle(out_path)

    return {
        "csvs": sum(len(v) for v in method_to_csvs.values()),
        "methods": len(method_to_csvs),
        "rows_used": total_rows_used,
        "status": "ok",
    }


def main() -> None:
    print(f"Rebuilding aggregated pkls from raw CSVs under {RAW_DIR}")
    print(f"Writing to {AGG_DIR}")
    print()
    summary = []
    for log in sorted(BENCH_LOGS):
        info = rebuild_log(log)
        summary.append({"log": log, **info})
        print(f"  {log:40s}  csvs={info['csvs']:4d}  "
              f"methods={info['methods']:3d}  "
              f"rows_used={info['rows_used']:5d}  {info['status']}")
    df = pd.DataFrame(summary)
    print()
    print("Totals:")
    print(f"  logs processed : {len(df)}")
    print(f"  logs OK        : {(df['status'] == 'ok').sum()}")
    print(f"  total CSVs     : {df['csvs'].sum()}")
    print(f"  total methods  : {df['methods'].sum()}")
    print(f"  total rows     : {df['rows_used'].sum()}")


if __name__ == "__main__":
    main()

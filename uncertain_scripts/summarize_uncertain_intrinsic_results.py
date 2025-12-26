"""
PyCharm-friendly summarizer for the *uncertain intrinsic evaluation* results.

What it does
------------
- Reads the per-run dfavg pickles written by:
  `evaluation/evaluation_of_activity_distances/intrinsic_evaluation_uncertain/evaluation_activity_distance_intrinsic_uncertain.py`
  under:
    results/activity_distances/intrinsic_uncertain_df_avg/<log_name>/u_<u>/dfavg_r<r>_w<w>_samplesize_<s>.pkl
- Filters to a user-provided list of logs, (r,w,s,u) settings, and methods.
- Writes a new CSV with one row per (log, r, w, s, u, method).
- Aggregates over logs (and over multiple r/w/s entries if provided) and writes a second CSV.
- Creates grouped bar charts per metric:
    - triplet
    - prec (= precision@w-1)
    - nn (= precision@1)
    - diameter
  Methods are sorted alphabetically; for each method, the bars for increasing u are adjacent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# Repo import shim (PyCharm-friendly)
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR


# =============================================================================
# Configuration (edit in IDE)
# =============================================================================

# List of log folder names (as used under results/activity_distances/intrinsic_uncertain_df_avg/<log_name>/)
LOG_NAMES = [
    # "ikea_asm__clip_based__i3d__dev2__pretrained__rgb__xes_uncertain_pred_merged",
]

# Intrinsic benchmark parameters to include
R_VALUES = [10]
W_VALUES = [5]
S_VALUES = [5]
U_VALUES = [1, 2, 3, 4, 5]

# If you pass base methods without window suffix, set WINDOW_SIZES and METHOD_BASES.
WINDOW_SIZES = [3, 5, 9]
METHOD_BASES = [
    # Uncertain count-based family (examples)
    "Uncertain AA Seq",
    "Uncertain AA Seq PMI",
    "Uncertain AA Seq PPMI",
    "Uncertain AA MSet",
    "Uncertain AA MSet PMI",
    "Uncertain AA MSet PPMI",
    "Uncertain AC Seq",
    "Uncertain AC Seq PMI",
    "Uncertain AC Seq PPMI",
    "Uncertain AC MSet",
    "Uncertain AC MSet PMI",
    "Uncertain AC MSet PPMI",
    # Uncertain act2vec
    "Uncertain act2vec CBOW",
    "Uncertain act2vec Skip-gram",
]

# If you want full control, set METHODS explicitly (must match dfavg["Distance Function"] exactly),
# and set METHOD_BASES = [].
METHODS: List[str] = []

# Output
OUT_DIR = Path(ROOT_DIR) / "results" / "activity_distances" / "intrinsic_uncertain_summary"
OUT_CSV_RAW = OUT_DIR / "intrinsic_uncertain_selected_rows.csv"
OUT_CSV_AGG = OUT_DIR / "intrinsic_uncertain_aggregated_mean.csv"
OUT_CSV_MISSING = OUT_DIR / "intrinsic_uncertain_missing_configs.csv"
OUT_PLOT_DIR = OUT_DIR / "plots"

# If True, print a short report of requested configs for which no dfavg pickle exists.
PRINT_MISSING = True

# If True, abort immediately when any requested dfavg pickle is missing.
STRICT = False


# =============================================================================
# Implementation
# =============================================================================


def _expand_methods() -> List[str]:
    if METHODS:
        return list(METHODS)
    out: List[str] = []
    for base in METHOD_BASES:
        for ws in WINDOW_SIZES:
            out.append(f"{base} w_{int(ws)}")
    return out


def _dfavg_path(*, log_name: str, u: int, r: int, w: int, s: int) -> Path:
    return (
        Path(ROOT_DIR)
        / "results"
        / "activity_distances"
        / "intrinsic_uncertain_df_avg"
        / log_name
        / f"u_{int(u)}"
        / f"dfavg_r{int(r)}_w{int(w)}_samplesize_{int(s)}.pkl"
    )


def _load_dfavg(path: Path) -> pd.DataFrame:
    # dfavg is a pickled pandas DataFrame (single row)
    return pd.read_pickle(path)


def collect_selected_rows(
    *,
    log_names: Sequence[str],
    r_values: Sequence[int],
    w_values: Sequence[int],
    s_values: Sequence[int],
    u_values: Sequence[int],
    methods: Sequence[str],
    strict: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []

    for log_name in log_names:
        for u in u_values:
            for r in r_values:
                for w in w_values:
                    for s in s_values:
                        pkl = _dfavg_path(log_name=log_name, u=u, r=r, w=w, s=s)
                        if not pkl.exists():
                            if strict:
                                raise FileNotFoundError(str(pkl))
                            missing.append(
                                {
                                    "log_name": log_name,
                                    "u": int(u),
                                    "r": int(r),
                                    "w": int(w),
                                    "s": int(s),
                                    "expected_file": str(pkl),
                                }
                            )
                            continue
                        df = _load_dfavg(pkl)
                        if df is None or len(df) == 0:
                            continue

                        # Normalize column names from deterministic/uncertain dfavg variants
                        # Expected columns: Log Name, Distance Function, u, diameter, precision@w-1, precision@1, triplet
                        df = df.copy()
                        if "Log Name" not in df.columns and "log_name" in df.columns:
                            df["Log Name"] = df["log_name"]
                        if "Distance Function" not in df.columns and "method" in df.columns:
                            df["Distance Function"] = df["method"]

                        for _idx, rec in df.iterrows():
                            if str(rec.get("Distance Function")) not in set(methods):
                                continue
                            rows.append(
                                {
                                    "log_name": log_name,
                                    "method": str(rec.get("Distance Function")),
                                    "u": int(rec.get("u")),
                                    "r": int(r),
                                    "w": int(w),
                                    "s": int(s),
                                    "avg_norm_entropy": float(rec.get("avg_norm_entropy", 0.0)),
                                    "diameter": float(rec.get("diameter", 0.0)),
                                    "prec": float(rec.get("precision@w-1", 0.0)),
                                    "nn": float(rec.get("precision@1", 0.0)),
                                    "triplet": float(rec.get("triplet", 0.0)),
                                    "source_file": str(pkl),
                                }
                            )

    if not rows:
        out = pd.DataFrame(
            columns=[
                "log_name",
                "method",
                "u",
                "r",
                "w",
                "s",
                "avg_norm_entropy",
                "diameter",
                "prec",
                "nn",
                "triplet",
                "source_file",
            ]
        )
        out.attrs["missing"] = missing
        return out
    out = pd.DataFrame(rows)
    out.attrs["missing"] = missing
    return out


def aggregate_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across logs and (r,w,s) settings: mean per (method, u).
    """
    if df.empty:
        return df.copy()
    g = df.groupby(["method", "u"], as_index=False)
    out = g.agg(
        n=("log_name", "count"),
        diameter=("diameter", "mean"),
        prec=("prec", "mean"),
        nn=("nn", "mean"),
        triplet=("triplet", "mean"),
        avg_norm_entropy=("avg_norm_entropy", "mean"),
    )
    return out


def plot_grouped_bars(
    df_agg: pd.DataFrame,
    *,
    metric: str,
    out_path: Path,
    title: Optional[str] = None,
    u_values: Optional[Sequence[int]] = None,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    if df_agg.empty:
        return
    if metric not in ("triplet", "prec", "nn", "diameter"):
        raise ValueError(f"Unknown metric: {metric}")

    methods = sorted(df_agg["method"].unique().tolist())
    u_vals = sorted(df_agg["u"].unique().tolist()) if u_values is None else [int(u) for u in u_values]

    # Build matrix values[method_idx, u_idx]
    val = np.full((len(methods), len(u_vals)), np.nan, dtype=float)
    idx_m = {m: i for i, m in enumerate(methods)}
    idx_u = {u: j for j, u in enumerate(u_vals)}
    for rec in df_agg.to_dict(orient="records"):
        m = rec["method"]
        u = int(rec["u"])
        if m in idx_m and u in idx_u:
            val[idx_m[m], idx_u[u]] = float(rec.get(metric, float("nan")))

    x = np.arange(len(methods))
    group_width = 0.8
    bar_width = group_width / max(1, len(u_vals))
    offsets = (np.arange(len(u_vals)) - (len(u_vals) - 1) / 2.0) * bar_width

    fig_w = max(10, int(0.5 * len(methods)))
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    for j, u in enumerate(u_vals):
        ax.bar(x + offsets[j], val[:, j], width=bar_width, label=f"u={u}")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.legend(ncol=min(len(u_vals), 5), fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    methods = _expand_methods()
    df_raw = collect_selected_rows(
        log_names=LOG_NAMES,
        r_values=R_VALUES,
        w_values=W_VALUES,
        s_values=S_VALUES,
        u_values=U_VALUES,
        methods=methods,
        strict=STRICT,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(OUT_CSV_RAW, index=False)

    missing = df_raw.attrs.get("missing", [])
    if missing:
        df_missing = pd.DataFrame(missing)
        df_missing.to_csv(OUT_CSV_MISSING, index=False)
        if PRINT_MISSING:
            # Keep console output short: show count + first N examples.
            head_n = min(20, len(df_missing))
            print(f"[summarize] missing dfavg files: {len(df_missing)} (showing first {head_n})")
            print(df_missing.head(head_n).to_string(index=False))
            print(f"[summarize] wrote missing list to: {OUT_CSV_MISSING}")
    else:
        # Still write an empty missing CSV for reproducibility.
        pd.DataFrame(columns=["log_name", "u", "r", "w", "s", "expected_file"]).to_csv(OUT_CSV_MISSING, index=False)

    df_agg = aggregate_mean(df_raw)
    df_agg.to_csv(OUT_CSV_AGG, index=False)

    # Plots
    OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_grouped_bars(df_agg, metric="triplet", out_path=OUT_PLOT_DIR / "triplet.png", title="Triplet (mean)")
    plot_grouped_bars(df_agg, metric="prec", out_path=OUT_PLOT_DIR / "prec.png", title="Precision@w-1 (mean)")
    plot_grouped_bars(df_agg, metric="nn", out_path=OUT_PLOT_DIR / "nn.png", title="Nearest-neighbor / Precision@1 (mean)")
    plot_grouped_bars(df_agg, metric="diameter", out_path=OUT_PLOT_DIR / "diameter.png", title="Diameter (mean)")

    print(f"Wrote: {OUT_CSV_RAW}")
    print(f"Wrote: {OUT_CSV_AGG}")
    print(f"Wrote: {OUT_CSV_MISSING}")
    print(f"Wrote plots to: {OUT_PLOT_DIR}")


if __name__ == "__main__":
    main()



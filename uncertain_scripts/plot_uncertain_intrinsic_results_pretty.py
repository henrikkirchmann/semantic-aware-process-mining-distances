"""
Paper-quality plots for the *uncertain intrinsic evaluation*.

Reads the CSV produced by `uncertain_scripts/summarize_uncertain_intrinsic_results.py`:
  results/activity_distances/intrinsic_uncertain_summary/intrinsic_uncertain_aggregated_mean.csv

Creates:
- 4 individual vector plots (PDF/SVG): I_comp, I_nn, I_prec, I_tri
- 1 combined 2x2 figure (PDF/SVG) with the same four plots

Plot design goals:
- Seaborn style, Times-family font (to match LaTeX)
- Methods sorted alphabetically
- For each method, bars for increasing u are placed next to each other
- Diameter is converted to I_comp = 1 - diameter (higher is better)
- Do not repeat "Uncertain" in method labels
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

IN_CSV = (
    Path(ROOT_DIR)
    / "results"
    / "activity_distances"
    / "intrinsic_uncertain_summary"
    / "intrinsic_uncertain_aggregated_mean.csv"
)

OUT_DIR = Path(ROOT_DIR) / "results" / "activity_distances" / "intrinsic_uncertain_summary" / "paper_plots"

# Export formats (vector graphics)
SAVE_PDF = True
SAVE_SVG = True

# If provided, restrict to these u values (in this order)
U_ORDER: Optional[List[int]] = [1, 2, 3, 4, 5]

# If True, rotate x tick labels (recommended for long method names)
ROTATE_XTICKS = True

# Figure sizing
FIG_W = 10.5
FIG_H = 3.2
COMBINED_FIG_W = 12.0
COMBINED_FIG_H = 8.0


# =============================================================================
# Helpers
# =============================================================================


def _pretty_method_name(method: str) -> str:
    """
    Make method labels paper-friendly:
    - drop leading "Uncertain "
    - keep window size suffix but as "w=3" (if present)
    """
    m = str(method)
    if m.startswith("Uncertain "):
        m = m[len("Uncertain ") :]
    m = m.replace(" w_", " (w=") + (")" if " w_" in method else "")
    # If we introduced "(w=" twice by the replace trick above, fix:
    m = m.replace("(w=", "w=") if m.count("(w=") > 1 else m
    # Normalize act2vec naming a bit
    m = m.replace("act2vec ", "act2vec ")
    return m


def _load_and_prepare(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        # Attempt auto-discovery (useful when results dir differs or is not tracked in git).
        candidates = list(Path(ROOT_DIR).rglob("intrinsic_uncertain_aggregated_mean.csv"))
        if candidates:
            path = candidates[0]
        else:
            raise FileNotFoundError(
                "Could not find the input CSV. Set IN_CSV to the path of "
                "`intrinsic_uncertain_aggregated_mean.csv` produced by "
                "`uncertain_scripts/summarize_uncertain_intrinsic_results.py`."
            )

    df = pd.read_csv(path)
    # Expected columns from summarizer: method,u,n,diameter,prec,nn,triplet,avg_norm_entropy
    required = {"method", "u", "diameter", "prec", "nn", "triplet"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)} (got {sorted(df.columns)})")

    df = df.copy()
    df["u"] = df["u"].astype(int)
    df["method_pretty"] = df["method"].map(_pretty_method_name)
    df["I_comp"] = 1.0 - df["diameter"].astype(float)
    df["I_nn"] = df["nn"].astype(float)
    df["I_prec"] = df["prec"].astype(float)
    df["I_tri"] = df["triplet"].astype(float)

    # Enforce order
    if U_ORDER is not None:
        df = df[df["u"].isin(set(U_ORDER))].copy()
        df["u"] = pd.Categorical(df["u"], categories=list(U_ORDER), ordered=True)
    else:
        df["u"] = pd.Categorical(df["u"], categories=sorted(df["u"].unique().tolist()), ordered=True)

    method_order = sorted(df["method_pretty"].unique().tolist())
    df["method_pretty"] = pd.Categorical(df["method_pretty"], categories=method_order, ordered=True)
    return df


def _setup_style() -> None:
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid")
    # Times-family for consistency with LaTeX (falls back if unavailable)
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
    # Vector text output
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"


def _save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PDF:
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")


def _plot_metric(df: pd.DataFrame, metric_col: str, ylabel: str, title: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.barplot(
        data=df,
        x="method_pretty",
        y=metric_col,
        hue="u",
        ax=ax,
        dodge=True,
        errorbar=None,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    if ROTATE_XTICKS:
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
    ax.legend(title="u", ncol=min(5, df["u"].nunique()), fontsize=9, title_fontsize=9, frameon=True)
    fig.tight_layout()
    return fig


def _plot_combined(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(COMBINED_FIG_W, COMBINED_FIG_H), sharex=True, sharey=False)
    axes = axes.ravel()

    specs = [
        ("I_comp", r"$I_{comp}$ (higher is better)", r"$I_{comp}$"),
        ("I_nn", r"$I_{nn}$", r"$I_{nn}$"),
        ("I_prec", r"$I_{prec}$", r"$I_{prec}$"),
        ("I_tri", r"$I_{tri}$", r"$I_{tri}$"),
    ]

    for ax, (col, ylabel, title) in zip(axes, specs):
        sns.barplot(
            data=df,
            x="method_pretty",
            y=col,
            hue="u",
            ax=ax,
            dodge=True,
            errorbar=None,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)

    # One shared legend (use the legend from the first axis)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()

    fig.legend(handles, labels, title="u", loc="lower center", ncol=min(5, len(labels)), frameon=True)

    if ROTATE_XTICKS:
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def main() -> None:
    _setup_style()
    df = _load_and_prepare(IN_CSV)

    # Individual plots
    fig = _plot_metric(df, "I_comp", r"$I_{comp}$ (higher is better)", r"Intrinsic evaluation ($I_{comp}=1-\mathrm{diameter}$)")
    _save(fig, "intrinsic_uncertain_I_comp")
    fig = _plot_metric(df, "I_nn", r"$I_{nn}$ (higher is better)", r"Intrinsic evaluation ($I_{nn}$)")
    _save(fig, "intrinsic_uncertain_I_nn")
    fig = _plot_metric(df, "I_prec", r"$I_{prec}$ (higher is better)", r"Intrinsic evaluation ($I_{prec}$)")
    _save(fig, "intrinsic_uncertain_I_prec")
    fig = _plot_metric(df, "I_tri", r"$I_{tri}$ (higher is better)", r"Intrinsic evaluation ($I_{tri}$)")
    _save(fig, "intrinsic_uncertain_I_tri")

    # Combined 2x2
    fig = _plot_combined(df)
    _save(fig, "intrinsic_uncertain_all_metrics_2x2")

    print(f"Read: {IN_CSV if Path(IN_CSV).exists() else 'auto-discovered'}")
    print(f"Wrote plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()



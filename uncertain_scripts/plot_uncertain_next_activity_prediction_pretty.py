"""
Paper-quality plots for the *uncertain next-activity prediction* benchmark (Evermann model).

Reads the per-run CSVs produced by:
  `uncertain_scripts/run_uncertain_next_activity_prediction_evermann.py`

Expected inputs (by default):
  results/next_activity_prediction_uncertain_evermann__clip_based__i3d__dev2__pretrained__rgb.csv
  results/next_activity_prediction_uncertain_evermann__pose_based__HCN_32__pretrained__pose__dev3.csv

Creates a single figure with two subplots (one per log / model_id) as vector graphics (PDF/SVG),
showing test accuracy (`test_acc`) as grouped bars per method.

Bar grouping logic
------------------
- Uncertainty level u is derived from `embedding_training`:
    top1_determinized -> u=1
    top3_uncertain    -> u=3
  (and generally: "top{k}_..." -> u=k if present)

- For non-AC methods, we compare two event representations:
    expected_embedding   -> "Expected"
    scaled_concat_full   -> "Scaled Concat"
  yielding up to 4 bars per method: (u in {1,3}) x (repr in {Expected, Scaled Concat})

- Baselines:
  Rows with `representation` in {argmax_onehot, weighted_onehot} are shown as separate x-axis
  categories and renamed to:
    "Argmax Onehot"
    "Weighted Onehot"
  These are plotted as a single "Baseline" bar (they do not participate in u×repr variants).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

IN_CSVS: List[Path] = [
    Path(ROOT_DIR) / "results" / "next_activity_prediction_uncertain_evermann__clip_based__i3d__dev2__pretrained__rgb.csv",
    Path(ROOT_DIR) / "results" / "next_activity_prediction_uncertain_evermann__pose_based__HCN_32__pretrained__pose__dev3.csv",
]

OUT_DIR = Path(ROOT_DIR) / "results" / "next_activity_prediction_uncertain_evermann" / "paper_plots"

# Export formats (vector graphics)
SAVE_PDF = True
SAVE_SVG = True

# If set, filter to these window sizes. If None, keep all.
WINDOW_SIZES: Optional[List[int]] = [3, 5]

# Only show these uncertainty levels (derived from embedding_training). If None, keep all.
U_LEVELS: Optional[List[int]] = [1, 3]

# Figure sizing
FIG_W = 13.5
FIG_H = 5.0

# Rotate x labels (recommended)
ROTATE_XTICKS = True

# Legend placement
LEGEND_LOC = "lower center"


# =============================================================================
# Helpers
# =============================================================================


def _setup_style() -> None:
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["mathtext.fontset"] = "stix"


def _pretty_method_name(method: str) -> str:
    import re

    m = str(method)
    if m.startswith("Uncertain "):
        m = m[len("Uncertain ") :]
    m = m.replace("act2vec Skip-gram", "act2vec SG")
    m = re.sub(r"\s+w_(\d+)\s*$", r" Win \1", m)  # just in case (legacy)
    return m


def _derive_u_from_embedding_training(embedding_training: str) -> Optional[int]:
    import re

    s = str(embedding_training or "")
    # Typical: top1_determinized / top3_uncertain / top2_uncertain
    m = re.match(r"^\s*top(\d+)\b", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _repr_short(repr_name: str) -> str:
    r = str(repr_name)
    if r == "expected_embedding":
        return "Expected"
    if r == "scaled_concat_full":
        return "Scaled Concat"
    if r == "argmax_onehot":
        return "Argmax Onehot"
    if r == "weighted_onehot":
        return "Weighted Onehot"
    return r


def _dataset_title(model_id: str) -> str:
    mid = str(model_id)
    if "clip_based__i3d" in mid and "rgb" in mid:
        return "IKEA ASM (RGB, I3D)"
    if "pose_based__HCN" in mid or "pose_based__hcn" in mid:
        return "IKEA ASM (Pose, HCN)"
    return mid


def _format_tick_label_with_bold_family(label: str) -> str:
    """
    Bold only the method family part (not the window suffix).
    """
    import re

    # Baselines: bold whole label
    if label in {"Argmax Onehot", "Weighted Onehot"}:
        lab = label.replace(" ", r"\;")
        return rf"$\mathbf{{{lab}}}$"

    fam_prefixes = (
        "AA MSet",
        "AA Seq",
        "AC MSet",
        "AC Seq",
        "act2vec CBOW",
        "act2vec SG",
    )
    m = re.match(r"^(.*?)(\s+Win\s+\d+)\s*$", str(label))
    if not m:
        return str(label)
    fam = m.group(1).strip()
    rest = m.group(2).strip()
    if not any(fam.startswith(pfx) for pfx in fam_prefixes):
        return str(label)
    fam_math = fam.replace(" ", r"\;")
    return rf"$\mathbf{{{fam_math}}}$ {rest}"


def _discover_csvs_if_missing(paths: List[Path]) -> List[Path]:
    existing = [p for p in paths if Path(p).exists()]
    if len(existing) == len(paths):
        return paths
    # Auto-discovery: pick newest matching files under ROOT_DIR/results/
    candidates = list(Path(ROOT_DIR).rglob("next_activity_prediction_uncertain_evermann__*.csv"))
    if not candidates:
        return paths
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer the two known model IDs if present; otherwise take the newest 2.
    wanted = [
        "next_activity_prediction_uncertain_evermann__clip_based__i3d__dev2__pretrained__rgb.csv",
        "next_activity_prediction_uncertain_evermann__pose_based__HCN_32__pretrained__pose__dev3.csv",
    ]
    out: List[Path] = []
    for w in wanted:
        for c in candidates:
            if c.name == w:
                out.append(c)
                break
    if len(out) == 2:
        return out
    return candidates[:2]


def _load_and_prepare(paths: List[Path]) -> pd.DataFrame:
    paths = _discover_csvs_if_missing(paths)

    frames: List[pd.DataFrame] = []
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(str(p))
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    required = {"model_id", "embedding_method", "window_size", "representation", "embedding_training", "test_acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)} (got {sorted(df.columns)})")

    df = df.copy()
    df["window_size"] = df["window_size"].astype(int)
    df["test_acc"] = df["test_acc"].astype(float)
    df["u"] = df["embedding_training"].map(_derive_u_from_embedding_training)
    df["repr_short"] = df["representation"].map(_repr_short)
    df["embedding_method_pretty"] = df["embedding_method"].map(_pretty_method_name)

    # Baselines: split out as separate x-axis categories
    is_baseline = df["representation"].isin(["argmax_onehot", "weighted_onehot"])
    df.loc[is_baseline, "method_pretty"] = df.loc[is_baseline, "repr_short"]
    df.loc[is_baseline, "variant"] = "Baseline"

    # Main methods: include window size in x-axis label (like intrinsic evaluation)
    df.loc[~is_baseline, "method_pretty"] = (
        df.loc[~is_baseline, "embedding_method_pretty"] + " Win " + df.loc[~is_baseline, "window_size"].astype(str)
    )
    df.loc[~is_baseline, "variant"] = (
        "u=" + df.loc[~is_baseline, "u"].astype("Int64").astype(str) + " " + df.loc[~is_baseline, "repr_short"]
    )

    # Filter window sizes / u levels (only affects non-baselines)
    if WINDOW_SIZES is not None:
        df = df[is_baseline | df["window_size"].isin(set(WINDOW_SIZES))].copy()
    if U_LEVELS is not None:
        df = df[is_baseline | df["u"].isin(set(U_LEVELS))].copy()

    # Baselines are duplicated for w=3 and w=5 in the results; collapse them.
    # If they ever differ, we average (and the plot still reflects a single bar).
    df = df.groupby(["model_id", "method_pretty", "variant"], as_index=False).agg(test_acc=("test_acc", "mean"))

    # Titles
    df["model_title"] = df["model_id"].map(_dataset_title)

    # Variant ordering (only keep those present)
    variant_order_all = [
        "u=1 Expected",
        "u=1 Scaled Concat",
        "u=3 Expected",
        "u=3 Scaled Concat",
        "Baseline",
    ]
    present = [v for v in variant_order_all if v in set(df["variant"].unique().tolist())]
    df["variant"] = pd.Categorical(df["variant"], categories=present, ordered=True)

    # X-axis ordering: group by base method, then by window size (Win 3, Win 5, ...), baselines last.
    def _split_base_win(s: str) -> Tuple[str, int]:
        import re

        m = str(s)
        mm = re.search(r"\sWin\s(\d+)\s*$", m)
        if mm:
            win = int(mm.group(1))
            base = m[: mm.start()].rstrip()
            return (base, win)
        return (m, 10**9)

    method_order = sorted(df["method_pretty"].unique().tolist(), key=_split_base_win)
    df["method_pretty"] = pd.Categorical(df["method_pretty"], categories=method_order, ordered=True)

    return df


def _save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PDF:
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")


def _plot_two_panel(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    titles = df[["model_id", "model_title"]].drop_duplicates().sort_values("model_title")
    model_ids = titles["model_id"].tolist()
    if len(model_ids) != 2:
        # Still plot whatever is present
        model_ids = df["model_id"].drop_duplicates().tolist()

    fig, axes = plt.subplots(1, len(model_ids), figsize=(FIG_W, FIG_H), sharey=True)
    if len(model_ids) == 1:
        axes = [axes]

    palette: Dict[str, str] = {
        "u=1 Expected": "#4C72B0",
        "u=1 Scaled Concat": "#55A868",
        "u=3 Expected": "#C44E52",
        "u=3 Scaled Concat": "#8172B3",
        "Baseline": "#4D4D4D",
    }

    for ax, mid in zip(axes, model_ids):
        sub = df[df["model_id"] == mid].copy()
        sns.barplot(
            data=sub,
            x="method_pretty",
            y="test_acc",
            hue="variant",
            ax=ax,
            dodge=True,
            errorbar=None,
            palette=palette,
        )
        # Remove bar borders
        for p in ax.patches:
            try:
                p.set_linewidth(0.0)
                p.set_edgecolor(p.get_facecolor())
            except Exception:
                pass

        # Tick label formatting (force set to preserve mathtext bolding)
        try:
            cats = [str(c) for c in sub["method_pretty"].cat.categories]
            ticks = ax.get_xticks()
            # Guard: only set labels if tick count matches category count (avoids Matplotlib warnings).
            if len(ticks) == len(cats):
                ax.set_xticks(ticks)
                ax.set_xticklabels([_format_tick_label_with_bold_family(c) for c in cats])
        except Exception:
            pass

        ax.set_xlabel("")
        ax.set_ylabel("Test Accuracy" if ax is axes[0] else "")
        ax.set_title(_dataset_title(mid))
        ax.set_ylim(0.0, 1.0)

        if ROTATE_XTICKS:
            ax.tick_params(axis="x", rotation=45)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")

        # separators between base methods (after each Win block)
        import re

        cats = [str(c) for c in sub["method_pretty"].cat.categories]
        bases = [re.sub(r"\sWin\s\d+\s*$", "", c).rstrip() for c in cats]
        last_idx_by_base: Dict[str, int] = {}
        for i, b in enumerate(bases):
            last_idx_by_base[b] = i
        last_indices = sorted(set(last_idx_by_base.values()))
        for idx in last_indices[:-1]:
            ax.axvline(idx + 0.5, color="0.5", linestyle=(0, (2, 3)), linewidth=0.8, alpha=0.6, zorder=0)

    # Shared legend (bottom center)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.legend(
        handles,
        labels,
        title="Setting",
        loc=LEGEND_LOC,
        ncol=max(1, min(5, len(labels))),
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def main() -> None:
    _setup_style()
    df = _load_and_prepare(IN_CSVS)
    fig = _plot_two_panel(df)
    _save(fig, "uncertain_next_activity_prediction_test_acc_2panel")
    print(f"Wrote plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()



"""
PyCharm-friendly runner for uncertain runtime evaluation.

Runs runtime measurements for uncertain activity similarity methods on selected IKEA ASM uncertain logs.
This calls the same implementation used by:
  `evaluation/evaluation_of_activity_distances/runtime_analysis/runtime_analysis.py --mode uncertain`

You can edit the constants below and run this file directly from an IDE.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR
from evaluation.evaluation_of_activity_distances.runtime_analysis.runtime_analysis import (
    evaluate_runtime_uncertain,
)
from evaluation.data_util.uncertain_evaluation_helpers import add_window_size_evaluation
from evaluation.data_util.util_activity_distances_uncertain import (
    UNCERTAIN_COUNT_BASED_METHODS,
    UNCERTAIN_NEURAL_METHODS,
)


# =============================================================================
# Configuration (edit in IDE)
# =============================================================================

LOG_LIST = [
    # ResNet-18 (RGB, dev3)
    "ikea_asm__frame_based__resnet18__pretrained__rgb__dev3__xes_uncertain_pred_merged",
    # ST-GCN-64 (Pose, dev3)
    "ikea_asm__pose_based__ST_GCN_64__pretrained__pose__dev3__xes_uncertain_pred_merged",
]

WINDOW_SIZES = [3, 5, 9]
REPETITIONS = 3

NA_LABEL = "NA"

# Intrinsic-benchmark semantics for runtime evaluation:
# apply top-u truncation per event *with renormalization* before running methods.
UNCERTAINTY_LEVEL_U = 3

# Optional debugging: only read the first N traces
LIMIT_TRACES = None


if __name__ == "__main__":
    methods = add_window_size_evaluation(
        list(UNCERTAIN_COUNT_BASED_METHODS) + list(UNCERTAIN_NEURAL_METHODS),
        WINDOW_SIZES,
    )
    print(f"[uncertain-runtime] logs={LOG_LIST}")
    print(f"[uncertain-runtime] methods={len(methods)} window_sizes={WINDOW_SIZES} repetitions={REPETITIONS}")

    results = evaluate_runtime_uncertain(
        methods,
        LOG_LIST,
        REPETITIONS,
        na_label=NA_LABEL,
        uncertainty_level_u=UNCERTAINTY_LEVEL_U,
        limit_traces=LIMIT_TRACES,
    )

    # Write a separate CSV so it's easy to find from the IDE.
    out = Path(ROOT_DIR) / "results" / f"runtime_results_uncertain_pycharm_{REPETITIONS}_repetitions.csv"
    import pandas as pd

    pd.DataFrame(results).to_csv(out, index=False)
    print(f"[uncertain-runtime] saved: {out}")



"""
Demo: window-based uncertain expected counts (no full trace realization enumeration).

This script exists to highlight the difference:
- trace-based (old): enumerate trace realizations (expensive)
- window-based (new): enumerate window realizations around each center (usually cheaper)
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR
from uncertain_utils.uncertain_xes_reader import read_uncertain_xes
from distances.uncertain_activity_distances.data_util.uncertain_window_based_expected_counts import (
    compute_expected_counts_window_based,
)


# =============================================================================
# Hardcoded config
# =============================================================================

XES_PATH = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"
LIMIT_TRACES = 3
WINDOW_SIZE = 5
CONTEXT_KIND = "seq"  # "seq" or "mset"
NA_LABEL = "NA"


if __name__ == "__main__":
    log = read_uncertain_xes(XES_PATH, limit_traces=LIMIT_TRACES)
    expected_counts, activity_freq = compute_expected_counts_window_based(
        log,
        window_size=WINDOW_SIZE,
        context_kind=CONTEXT_KIND,
        na_label=NA_LABEL,
        progress=lambda m: print(m, flush=True),
    )
    print("done activities", len(activity_freq), "contexts_for_first_activity", len(next(iter(expected_counts.values()))) if expected_counts else 0)




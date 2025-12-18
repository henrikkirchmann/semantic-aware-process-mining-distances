"""
Run uncertain count-based activity distances (PyCharm-friendly).

This script is intentionally *not* CLI-driven: change the constants below and run.
It demonstrates the new uncertain log reader + uncertain count-based AA/AC methods.

Typical workflow
----------------
1) Put an uncertain XES into the repo (e.g., `xes_uncertain_gt__no_na.xes`)
2) Update `XES_PATH` below (keep it relative to ROOT_DIR for portability)
3) Choose `METHOD_NAME` and `WINDOW_SIZE`
4) Run the script in an IDE
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

# Ensure repo root is on PYTHONPATH when running this file directly (outside an IDE).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances_uncertain import (
    UNCERTAIN_COUNT_BASED_METHODS,
    get_uncertain_activity_distance_matrix,
)
from uncertain_utils.uncertain_xes_reader import read_uncertain_xes


# =============================================================================
# Hardcoded configuration (edit these in PyCharm)
# =============================================================================

XES_PATH = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"

# Quick debugging knobs
LIMIT_TRACES = 25  # set to None for full file

# Choose one of:
#   evaluation.data_util.util_activity_distances_uncertain.UNCERTAIN_COUNT_BASED_METHODS
METHOD_NAME = "Uncertain AC Seq PPMI"

WINDOW_SIZE = 5  # must be odd: 3,5,9,...

# The IKEA ASM exports store *dense* probs_json with many tiny values.
# If you want to consider *all possible traces* (exact), you must not prune:
#   TOP_K = None and MIN_PROB = 0.0
# Warning: exact trace-realization enumeration is exponential in trace length.
TOP_K = None              # keep only top-k activities per event (set to None for exact)
MIN_PROB = 0.0            # additionally drop probs below this

# NA semantics for uncertain traces: if an event is realized as NA, the event is removed
# from the deterministic trace (i.e., "did not happen").
NA_LABEL = "NA"

# Debug safety valve: cap the number of enumerated deterministic realizations per uncertain trace.
# Set to None for exact enumeration (can explode combinatorially).
#
# Tip: for exact mode, the only reliable way to keep runtime manageable is to reduce LIMIT_TRACES
# and/or work with logs/traces where each event distribution has small support.
MAX_REALIZATIONS_PER_TRACE = None

# Progress printing
PRINT_PROGRESS = True
PROGRESS_EVERY_REALIZATIONS = 50_000

# If you kept NA events, you may want to exclude them here (your `__no_na` file should not contain NA).
EXCLUDE_ACTIVITIES = set()

# Nearest-neighbor demo
QUERY_ACTIVITY = None     # e.g., "tighten leg" or set to None to auto-pick
K_NEIGHBORS = 10


def _nested_from_pair_dict(d):
    nested = defaultdict(dict)
    for (a, b), v in d.items():
        nested[a][b] = v
    return dict(nested)


if __name__ == "__main__":
    print(f"Loading uncertain XES: {XES_PATH}")
    log = read_uncertain_xes(
        XES_PATH,
        drop_activities=None,
        min_prob=0.0,          # keep raw probs; pruning happens later
        renormalize=True,
        limit_traces=LIMIT_TRACES,
    )

    n_traces = len(log.traces)
    n_events = sum(len(t.events) for t in log.traces)
    print(f"log_name={log.log_name!r} traces={n_traces} events={n_events}")

    print("\nAvailable uncertain count-based methods:")
    for m in UNCERTAIN_COUNT_BASED_METHODS:
        print(f" - {m}")

    print("\nComputing distance matrix...")
    def _safe_progress(msg: str) -> None:
        try:
            print(msg, flush=True)
        except BrokenPipeError:
            # When piping output (e.g., to `head`), stdout may close early.
            # Progress output should never crash the computation.
            pass

    distance_matrix, debug = get_uncertain_activity_distance_matrix(
        log,
        method_name=METHOD_NAME,
        window_size=WINDOW_SIZE,
        top_k=TOP_K,
        min_prob=MIN_PROB,
        na_label=NA_LABEL,
        max_realizations_per_trace=MAX_REALIZATIONS_PER_TRACE,
        exclude_activities=EXCLUDE_ACTIVITIES,
        progress=_safe_progress if PRINT_PROGRESS else None,
        progress_every_realizations=PROGRESS_EVERY_REALIZATIONS,
    )

    # Pick a query activity
    if QUERY_ACTIVITY is None:
        # Choose the most frequent activity under the expected-count model
        act_freq = debug.get("activity_freq", {})
        if act_freq:
            QUERY_ACTIVITY = max(act_freq.items(), key=lambda kv: kv[1])[0]

    if QUERY_ACTIVITY is None:
        raise RuntimeError("Could not select a query activity (no activities found).")

    nested = _nested_from_pair_dict(distance_matrix)
    if QUERY_ACTIVITY not in nested:
        raise RuntimeError(f"Query activity {QUERY_ACTIVITY!r} not found. Available: {list(nested.keys())[:20]} ...")

    # Cosine distance: smaller is more similar
    neighbors = sorted(
        ((b, d) for b, d in nested[QUERY_ACTIVITY].items() if b != QUERY_ACTIVITY),
        key=lambda kv: kv[1],
    )[:K_NEIGHBORS]

    print(f"\nMethod: {METHOD_NAME}  (window={WINDOW_SIZE}, top_k={TOP_K}, na_label={NA_LABEL!r}, max_real={MAX_REALIZATIONS_PER_TRACE})")
    print(f"Nearest neighbors for {QUERY_ACTIVITY!r}:")
    for b, d in neighbors:
        print(f"  {b!r:40s}  dist={d:.4f}")



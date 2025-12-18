"""
IKEA uncertain log test run (with progress + feasibility estimate).

This file is meant to be run from an IDE (hardcoded config at the top).
It does two things:

1) Prints a per-trace *upper bound* on the number of deterministic trace realizations
   (product of per-event support sizes). This is a quick "feasibility check".
2) Runs one uncertain count-based method with progress output.

Why upper bound?
---------------
When NA is treated as "event absent", different label selections can collapse to
the same realized sequence; computing the exact number of distinct realizations is
hard. The product bound is cheap and still explains why exact enumeration explodes.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

# Ensure repo root is on PYTHONPATH when running this file directly (outside an IDE).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances_uncertain import (
    UNCERTAIN_COUNT_BASED_METHODS,
    get_uncertain_activity_distance_matrix,
)
from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import prune_distribution
from uncertain_utils.uncertain_xes_reader import read_uncertain_xes


# =============================================================================
# Hardcoded configuration
# =============================================================================

XES_PATH = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"
LIMIT_TRACES = None  # None = read full log

METHOD_NAME = "Uncertain AC Seq PPMI"
WINDOW_SIZE = 5

NA_LABEL = "NA"

# Exact mode (all possible traces): TOP_K=None, MIN_PROB=0.0, MAX_REALIZATIONS_PER_TRACE=None
TOP_K = None
MIN_PROB = 0.0
MAX_REALIZATIONS_PER_TRACE = None

PROGRESS_EVERY_REALIZATIONS = 50_000

# Run indefinitely by default (stop manually in your IDE).
STOP_AFTER_SECONDS = None


def safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    print(f"Loading uncertain XES: {XES_PATH}")
    log = read_uncertain_xes(XES_PATH, limit_traces=LIMIT_TRACES)
    print(f"log_name={log.log_name!r} traces={len(log.traces)}")

    print("\nAvailable methods:")
    for m in UNCERTAIN_COUNT_BASED_METHODS:
        print(f" - {m}")

    print("\nFeasibility (upper-bound #realizations per trace):")
    for i, tr in enumerate(log.traces, start=1):
        supports = []
        for ev in tr.events:
            opts = prune_distribution(ev.activity_probs, top_k=TOP_K, min_prob=MIN_PROB)
            supports.append(len(opts) if opts else 0)
        ub = 1
        for s in supports:
            ub *= max(1, s)
        print(f" - trace {i}: events={len(tr.events)} avg_support={sum(supports)/len(supports):.1f} ub={ub}")

    print("\nRunning method (with progress):")
    t0 = time.time()
    def progress(msg: str) -> None:
        if STOP_AFTER_SECONDS is not None and (time.time() - t0) > STOP_AFTER_SECONDS:
            raise SystemExit(f"Stopped after {STOP_AFTER_SECONDS}s (demo-only). Set STOP_AFTER_SECONDS=None for a real exact run.")
        safe_print(msg)

    dist, debug = get_uncertain_activity_distance_matrix(
        log,
        method_name=METHOD_NAME,
        window_size=WINDOW_SIZE,
        top_k=TOP_K,
        min_prob=MIN_PROB,
        na_label=NA_LABEL,
        max_realizations_per_trace=MAX_REALIZATIONS_PER_TRACE,
        progress=progress,
        progress_every_realizations=PROGRESS_EVERY_REALIZATIONS,
    )

    # Print a tiny sanity result
    acts = sorted({a for (a, _) in dist.keys()})
    print(f"\nComputed distances for |A|={len(acts)} activities. Example: dist[(a,a)] for first 5:")
    for a in acts[:5]:
        print(f"  {a!r}: {dist[(a,a)]:.4f}")



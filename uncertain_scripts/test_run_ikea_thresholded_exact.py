"""
IKEA test: thresholded probs (>=1%) + exact trace realization enumeration (no caps).

This matches the setup you asked for:
- keep only activities with probability >= 1% (done by pre-processing the XES)
- enumerate ALL trace realizations (no other limitations)

Important reality check:
Even after thresholding, exact enumeration can still be astronomically large.
This script prints an upper-bound estimate per trace and then starts the run
with progress output so you can observe throughput and stop manually in an IDE.
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
from uncertain_utils.uncertain_xes_probability_thresholding import ThresholdingConfig, threshold_xes_probs_json


# =============================================================================
# Hardcoded configuration
# =============================================================================

INPUT_XES_PATH = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"
THRESHOLD = 0.05  # keep only p >= 5% per event and renormalize
_thr_str = f"{THRESHOLD:.2f}".replace(".", "_")
XES_PATH = Path(ROOT_DIR) / "uncertain_event_logs" / f"xes_uncertain_gt__no_na__p_ge_{_thr_str}.xes"

LIMIT_TRACES = None  # None = full log

METHOD_NAME = "Uncertain AC Seq PPMI"
WINDOW_SIZE = 5

# Exact enumeration settings:
TOP_K = None
MIN_PROB = 0.0
MAX_REALIZATIONS_PER_TRACE = None

NA_LABEL = "NA"
PROGRESS_EVERY_REALIZATIONS = 50_000

# Run indefinitely by default (stop manually in your IDE).
STOP_AFTER_SECONDS = None


def safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    if not XES_PATH.exists():
        print(f"Thresholded XES not found, creating it now:")
        print(f"  input : {INPUT_XES_PATH}")
        print(f"  output: {XES_PATH}")
        cfg = ThresholdingConfig(threshold=THRESHOLD, renormalize=True)
        events_seen, events_updated = threshold_xes_probs_json(
            input_xes=INPUT_XES_PATH,
            output_xes=XES_PATH,
            config=cfg,
        )
        print(f"Done. events_seen={events_seen} events_updated={events_updated}")

    print(f"Loading thresholded XES: {XES_PATH}")
    log = read_uncertain_xes(XES_PATH, limit_traces=LIMIT_TRACES)
    print(f"log_name={log.log_name!r} traces={len(log.traces)}")

    print("\nAvailable methods:")
    for m in UNCERTAIN_COUNT_BASED_METHODS:
        print(f" - {m}")

    # Feasibility estimate: product of per-event support sizes under current pruning settings.
    print("\nFeasibility (upper-bound #realizations per trace):")
    for i, tr in enumerate(log.traces, start=1):
        supports = []
        for ev in tr.events:
            opts = prune_distribution(ev.activity_probs, top_k=TOP_K, min_prob=MIN_PROB)
            supports.append(len(opts) if opts else 0)
        ub = 1
        for s in supports:
            ub *= max(1, s)
        avg_support = (sum(supports) / len(supports)) if supports else 0.0
        print(f" - trace {i}: events={len(tr.events)} avg_support={avg_support:.1f} ub={ub}")

    print("\nRunning method (exact enumeration, with progress):")
    t0 = time.time()

    def progress(msg: str) -> None:
        if STOP_AFTER_SECONDS is not None and (time.time() - t0) > STOP_AFTER_SECONDS:
            raise SystemExit(
                f"Stopped after {STOP_AFTER_SECONDS}s (demo-only). Set STOP_AFTER_SECONDS=None for a real exact run."
            )
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

    acts = sorted({a for (a, _) in dist.keys()})
    print(f"\nComputed distances for |A|={len(acts)} activities.")



"""
IKEA test (thresholded) with multiprocessing across traces.

This uses process-based parallelism (not threads) to work around the Python GIL.
Each worker processes different traces, then we merge the partial counts.

Notes
-----
- On macOS, multiprocessing default start method is often "spawn", which can be slower due to pickling.
  If you run this in an IDE, using "fork" is typically faster for read-only workloads.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on PYTHONPATH when running directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances_uncertain import get_uncertain_activity_distance_matrix
from uncertain_utils.uncertain_xes_reader import read_uncertain_xes
from uncertain_utils.uncertain_xes_probability_thresholding import ThresholdingConfig, threshold_xes_probs_json


# =============================================================================
# Hardcoded configuration
# =============================================================================

INPUT_XES_PATH = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"
THRESHOLD = 0.05
_thr_str = f"{THRESHOLD:.2f}".replace(".", "_")
XES_PATH = Path(ROOT_DIR) / "uncertain_event_logs" / f"xes_uncertain_gt__no_na__p_ge_{_thr_str}.xes"
# Use the full log by default
LIMIT_TRACES = None

METHOD_NAME = "Uncertain AC Seq PPMI"
WINDOW_SIZE = 5

NA_LABEL = "NA"

# Exact enumeration per trace (still potentially huge!)
TOP_K = None
MIN_PROB = 0.0
# No further limitations on realizations (exact). Warning: can be extremely slow.
# For debugging/quick runs, set this to e.g. 2000.
MAX_REALIZATIONS_PER_TRACE = None

# Multiprocessing
N_JOBS = 8
MP_START_METHOD = "fork"  # try "fork" on macOS for speed; fall back to "spawn" if needed


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

    log = read_uncertain_xes(XES_PATH, limit_traces=LIMIT_TRACES)
    print(f"Loaded traces={len(log.traces)} from {XES_PATH}")
    dist, dbg = get_uncertain_activity_distance_matrix(
        log,
        method_name=METHOD_NAME,
        window_size=WINDOW_SIZE,
        top_k=TOP_K,
        min_prob=MIN_PROB,
        na_label=NA_LABEL,
        max_realizations_per_trace=MAX_REALIZATIONS_PER_TRACE,
        progress=safe_print,
        n_jobs=N_JOBS,
        mp_start_method=MP_START_METHOD,
    )
    print("done pairs", len(dist), "acts", len(dbg.get("activity_freq", {})))



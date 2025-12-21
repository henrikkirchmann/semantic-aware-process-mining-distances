"""
IDE-friendly runner: Uncertain Act2Vec (CBOW / Skip-gram) trained on window realizations.

This script runs the *default* uncertain act2vec mode implemented in:
  `distances/uncertain_activity_distances/uncertain_act2vec/uncertain_act2vec.py`

Key points
----------
- We train on *discrete window realizations* weighted by their probability mass (expected-loss training),
  not on soft labels.
- This matches the stochastically-known log semantics (one activity occurs, or the event is absent).
- For dense per-event distributions, exact enumeration can still be large; see `prob_threshold`/pruning
  options inside the config in `evaluation/data_util/util_activity_distances_uncertain.py` if needed.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances_uncertain import get_uncertain_activity_distance_matrix
from uncertain_utils.uncertain_xes_reader import read_uncertain_xes


# =============================================================================
# Hardcoded configuration (edit in IDE)
# =============================================================================

XES_PATH = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"
LIMIT_TRACES = None  # None = full log

# Choose one:
METHOD_NAME = "Uncertain act2vec CBOW"
# METHOD_NAME = "Uncertain act2vec Skip-gram"

WINDOW_SIZE = 3

# Optional pruning for dense event distributions (passed through to the method config):
TOP_K = None
MIN_PROB = 0.0

NA_LABEL = "NA"


def safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    safe_print(f"Loading uncertain XES: {XES_PATH}")
    log = read_uncertain_xes(XES_PATH, limit_traces=LIMIT_TRACES)
    safe_print(f"log_name={log.log_name!r} traces={len(log.traces)}")

    safe_print(f"Training: {METHOD_NAME} (window={WINDOW_SIZE})")
    dist, dbg = get_uncertain_activity_distance_matrix(
        log,
        method_name=METHOD_NAME,
        window_size=WINDOW_SIZE,
        top_k=TOP_K,
        min_prob=MIN_PROB,
        na_label=NA_LABEL,
        progress=safe_print,
    )

    safe_print(f"Done. pairs={len(dist)} activities={len(dbg.get('alphabet', dbg.get('activity_freq', {})))}")



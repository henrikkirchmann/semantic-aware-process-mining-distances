"""
PyCharm-friendly runner: Uncertain next-activity prediction (Evermann) on IKEA ASM.

Edits:
- Choose a model_id (action recognition model) and an embedding method.
- Choose representation:
    - "expected_embedding" (uses embedding_method)
    - "argmax_onehot"
    - "weighted_onehot"

Benchmark specifics:
- Target: first non-NA GT label after each predicted segment ends
- Input: predicted segments with avg_probs_json (top-3 capped per segment)
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluation_of_activity_distances.next_activity_prediction.next_activity_prediction_uncertain_evermann import (
    run_uncertain_next_activity_prediction,
)


# =============================================================================
# Configuration (edit in IDE)
# =============================================================================

MODEL_ID = "frame_based__resnet18__pretrained__rgb__dev3"
# MODEL_ID = "pose_based__ST_GCN_64__pretrained__pose__dev3"

WINDOW_SIZE = 3

# Embedding method only used for representation="expected_embedding"
EMBEDDING_METHOD = "Uncertain AA Seq"  # or "Uncertain AC Seq PMI", or "Uncertain act2vec CBOW", etc.

REPRESENTATION = "expected_embedding"
# REPRESENTATION = "argmax_onehot"
# REPRESENTATION = "weighted_onehot"

TOP_K_EVENT = 3  # cap per-segment distributions to top-3 labels before training embeddings / predictor
NA_LABEL = "NA"

# How to train embeddings (only used when REPRESENTATION="expected_embedding"):
# - "top3_uncertain": train embeddings on top-3 capped uncertain segment distributions (default, our method)
# - "top1_determinized": determinize each segment to its most likely non-NA label, then train embeddings
EMBEDDING_TRAINING = "top3_uncertain"
# EMBEDDING_TRAINING = "top1_determinized"

MAX_LEN = 20
SEED = 42

EPOCHS = 50
BATCH_SIZE = 64


if __name__ == "__main__":
    res = run_uncertain_next_activity_prediction(
        model_id=MODEL_ID,
        embedding_method=EMBEDDING_METHOD,
        window_size=WINDOW_SIZE,
        representation=REPRESENTATION,
        seed=SEED,
        max_len=MAX_LEN,
        top_k_event=TOP_K_EVENT,
        na_label=NA_LABEL,
        embedding_training=EMBEDDING_TRAINING,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    print(res)



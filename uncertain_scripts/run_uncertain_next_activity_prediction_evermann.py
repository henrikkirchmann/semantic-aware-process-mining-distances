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
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# =============================================================================
# Configuration (edit in IDE)
# =============================================================================

# Evaluate multiple IKEA ASM model outputs (like the intrinsic benchmark loops over logs).
# Each entry must match a folder name under:
#   uncertain_event_data/ikea_asm/split=test/model=<MODEL_ID>/
MODEL_IDS = [
    "frame_based__resnet18__pretrained__rgb__dev3",
    # "clip_based__i3d__dev2__pretrained__rgb",
    # "pose_based__ST_GCN_64__pretrained__pose__dev3",
]

# TensorFlow device selection:
# - "auto": let TensorFlow use GPU if available (requires compatible CUDA/cuDNN on the machine)
# - "cpu": force CPU (useful on clusters with incompatible CUDA/cuDNN runtimes)
TF_DEVICE = "auto"  # "auto" | "cpu"

# IMPORTANT: if we force CPU, do it BEFORE importing any module that might import TensorFlow.
# Note: this disables CUDA for the whole process (including PyTorch).
if TF_DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from evaluation.evaluation_of_activity_distances.next_activity_prediction.next_activity_prediction_uncertain_evermann import (
    run_uncertain_next_activity_prediction,
)

# Data root: directory that contains `uncertain_event_data/`.
# - Default (None) uses the repo root inferred inside the benchmark module.
# - On Windows, you can set this explicitly if you run the script from a different cwd/env.
DATA_ROOT = REPO_ROOT

# Which input representation(s) to run:
# - "expected_embedding": trains embeddings, then uses expected embedding per segment
# - "argmax_onehot": baseline (determinize each segment, one-hot)
# - "weighted_onehot": baseline (renormalized probabilities, one-hot weights)
REPRESENTATIONS = [
    "expected_embedding",
     "scaled_concat_full",  # full-alphabet concatenation: [p(a1)v(a1) || ... || p(a|A|)v(a|A|)]
     "argmax_onehot",
     "weighted_onehot",
]

# Embedding methods to sweep (used for embedding-based representations, e.g. "expected_embedding", "scaled_concat_full")
EMBEDDING_METHODS = [
    "Uncertain AA Seq",
     "Uncertain AA Seq PMI",
     "Uncertain AA Seq PPMI",
     "Uncertain AC Seq",
     "Uncertain AC Seq PMI",
     "Uncertain AC Seq PPMI",
     "Uncertain act2vec CBOW",
     "Uncertain act2vec Skip-gram",
]

# Window sizes to sweep (same role as in intrinsic evaluation)
WINDOW_SIZES = [3, 5]

TOP_K_EVENT = 3  # cap per-segment distributions to top-3 labels before training embeddings / predictor
NA_LABEL = "NA"

# How to train embeddings (only used when representation="expected_embedding"):
# - "top3_uncertain": train embeddings on top-3 capped uncertain segment distributions (default, our method)
# - "top2_uncertain": train embeddings on top-2 capped uncertain segment distributions
# - "top1_determinized": determinize each segment to its most likely non-NA label, then train embeddings
EMBEDDING_TRAININGS = [
    "top3_uncertain",
    # "top2_uncertain",
     "top1_determinized",
]

MAX_LEN = 20
SEED = 42
SPLIT_STRATEGY = "shuffle_seeded"  # "shuffle_seeded" (deterministic given SEED) | "sorted" (no shuffle)

EPOCHS = 50
BATCH_SIZE = 64

SAVE_RESULTS = True
RESULTS_DIR = REPO_ROOT / "results"

# Resume/skip behavior:
# - If OUT_CSV exists, we'll skip runs that are already present (by key) and append only missing runs.
# - Set to False if you want to re-run everything and overwrite OUT_CSV from scratch.
RESUME_IF_EXISTS = True

# Per-model outputs are defined inside the main loop below.
WRITE_SNAPSHOT_COPY = True


def _should_run_expected_embedding(representation: str) -> bool:
    # Representations that require embedding training / an embedding method sweep.
    return representation in ("expected_embedding", "scaled_concat_full")


def _iter_configs() -> List[Dict[str, Any]]:
    """
    Build a grid of run configurations.

    For one-hot baselines, embedding_method/window_size/embedding_training are not used by the model
    but we still record them for consistency (set to None).
    """
    runs: List[Dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        if _should_run_expected_embedding(representation):
            for embedding_method in EMBEDDING_METHODS:
                for window_size in WINDOW_SIZES:
                    for embedding_training in EMBEDDING_TRAININGS:
                        runs.append(
                            dict(
                                representation=representation,
                                embedding_method=embedding_method,
                                window_size=int(window_size),
                                embedding_training=embedding_training,
                            )
                        )
        else:
            # Baseline: doesn't train embeddings
            for window_size in WINDOW_SIZES:
                runs.append(
                    dict(
                        representation=representation,
                        embedding_method=None,
                        window_size=int(window_size),
                        embedding_training=None,
                    )
                )
    return runs


def _save_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    import pandas as pd

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)


def _run_key_from_row(row: Dict[str, Any]) -> tuple:
    """
    Key used for skipping already-computed runs.

    We include dataset/model + representation + embedding config + core benchmark config.
    """
    return (
        row.get("model_id"),
        row.get("representation"),
        row.get("top_k_event"),
        row.get("na_label"),
        row.get("max_len"),
        row.get("seed"),
        row.get("epochs"),
        row.get("batch_size"),
        # grid config (the intended sweep params; can be None for baselines)
        row.get("grid_embedding_method"),
        row.get("grid_embedding_training"),
        row.get("grid_window_size"),
    )


def _load_existing_keys(path: Path) -> set:
    import pandas as pd

    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if df.empty:
        return set()
    keys = set()
    for rec in df.to_dict(orient="records"):
        keys.add(_run_key_from_row(rec))
    return keys


def _append_rows_to_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    """
    Append rows to CSV, creating it if missing.
    """
    import pandas as pd

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not out_csv.exists():
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return
    df_new = pd.DataFrame(rows)
    df_new.to_csv(out_csv, index=False, mode="a", header=False)


if __name__ == "__main__":
    all_cfgs = _iter_configs()

    for MODEL_ID in MODEL_IDS:
        out_csv = RESULTS_DIR / f"next_activity_prediction_uncertain_evermann__{MODEL_ID}.csv"
        snapshot_csv = RESULTS_DIR / (
            f"next_activity_prediction_uncertain_evermann__{MODEL_ID}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        existing_keys = _load_existing_keys(out_csv) if (SAVE_RESULTS and RESUME_IF_EXISTS) else set()

        rows: List[Dict[str, Any]] = []  # rows computed in this run (for snapshot + printing)
        appended: List[Dict[str, Any]] = []  # rows to append to out_csv

        for cfg in all_cfgs:
            # For baselines, pass placeholders (function signature requires them).
            embedding_method = cfg["embedding_method"] or "Uncertain AA Seq"
            embedding_training = cfg["embedding_training"] or "top3_uncertain"
            window_size = int(cfg["window_size"])
            representation = str(cfg["representation"])

            res = run_uncertain_next_activity_prediction(
                model_id=MODEL_ID,
                embedding_method=str(embedding_method),
                window_size=window_size,
                representation=representation,
                seed=SEED,
                split_strategy=SPLIT_STRATEGY,
                max_len=MAX_LEN,
                top_k_event=TOP_K_EVENT,
                na_label=NA_LABEL,
                embedding_training=str(embedding_training),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                data_root=DATA_ROOT,
                force_tf_cpu=(TF_DEVICE == "cpu"),
            )
            # Keep the "grid" config explicit in the output row as well.
            res["grid_embedding_method"] = cfg["embedding_method"]
            res["grid_embedding_training"] = cfg["embedding_training"]
            res["grid_window_size"] = cfg["window_size"]
            # Also record TF training params for the run key.
            res["epochs"] = int(EPOCHS)
            res["batch_size"] = int(BATCH_SIZE)

            k = _run_key_from_row(res)
            if SAVE_RESULTS and RESUME_IF_EXISTS and k in existing_keys:
                print({"skipped": True, **res})
                continue

            rows.append(res)
            appended.append(res)
            existing_keys.add(k)
            print(res)

        if SAVE_RESULTS:
            if appended:
                _append_rows_to_csv(appended, out_csv)
            print(f"\nOut CSV: {out_csv}")
            print(f"Appended runs: {len(appended)}")
            print(f"Skipped runs: {len(all_cfgs) - len(appended)}")

            if WRITE_SNAPSHOT_COPY:
                _save_csv(rows, snapshot_csv)
                print(f"Snapshot CSV (this run only): {snapshot_csv}")



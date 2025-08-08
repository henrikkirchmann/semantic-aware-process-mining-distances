"""
Embedding memory usage reporter
================================

Purpose
-------
Compute selected activity embeddings for a given XES log and report the RAM
usage of the resulting embedding data structure. This focuses on the memory of
the final in-memory object (e.g., dict[str -> np.ndarray]) rather than
process-wide RSS over time.

How memory is measured
----------------------
- Deep object graph size ("deep_object_graph_size_bytes"): recursive traversal
  using a safe getsizeof-like approach (following referents) to account for
  Python container overhead and metadata.
- Numpy data buffers ("numpy_arrays_nbytes"): sum of np.ndarray.nbytes for all
  embedding vectors. This captures the actual contiguous memory of arrays.
- Estimated total ("estimated_total_bytes"): deep object graph size + numpy
  nbytes. This provides a practical upper-bound estimate of the embedding
  object's RAM footprint.

Limitations
-----------
- If numpy arrays share memory (views), their nbytes may be double-counted.

Usage in IDE (no CLI)
---------------------
1) Leave `USE_CLI = False`.
2) Edit the `USER_CONFIG` dict at the top of the file:
   - logs: ["Sepsis"]
   - methods: [] to use defaults, or e.g. ["Unit Distance", "De Koninck 2018 act2vec CBOW w_3"]
   - include_gamallo: True/False
   - window_sizes: e.g. [3, 5, 9]
   - output: CSV path for results
3) Run the file from your IDE.

CSV columns
-----------
- timestamp, log, method, num_activities, embedding_dim,
  shallow_dict_size_mb, keys_size_mb, values_overhead_mb,
  numpy_arrays_overhead_mb, numpy_arrays_nbytes_mb,
  deep_object_graph_size_mb, estimated_total_mb

Supported method names
----------------------
- "Unit Distance"
- "Bose 2009 Substitution Scores"
- "De Koninck 2018 act2vec CBOW" | "De Koninck 2018 act2vec skip-gram"
- "Activity-Activitiy Co Occurrence Bag Of Words" | PMI | PPMI
- "Activity-Context Bag Of Words" | N-Grams | PMI | PPMI
- "Chiorrini 2022 Embedding Process Structure"
- "Gamallo Fernandez 2023 Context Based" (heavy; optional)

Add a window suffix like "w_3" for methods that support window sizes.
"""
import argparse
import csv
import os
import sys
import re
import gc
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer

# Make project root importable
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from definitions import ROOT_DIR

# =====================
# In-file configuration
# Flip USE_CLI to True if you want to control via command line again.
USE_CLI: bool = False
USER_CONFIG = {
    "logs": ["Sepsis"],
    # Example: ["Unit Distance"], or [] to use default_methods
    "methods": ["Unit Distance"],
    "include_gamallo": False,
    "window_sizes": [3, 5, 9],
    "output": os.path.join(
        ROOT_DIR, "results", f"embedding_memory_usage_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ),
}
# =====================


def extract_window_size(text: str) -> int:
    match = re.search(r"w_(\d+)", text)
    return int(match.group(1)) if match else 3


def get_alphabet(log: List[List[str]]) -> List[str]:
    return sorted({activity for trace in log for activity in trace})


def get_obj_size(obj) -> int:
    if obj is None:
        return 0
    marked = {id(obj)}
    queue = [obj]
    total = 0
    while queue:
        total += sum(map(sys.getsizeof, queue))
        referents = ((id(o), o) for o in gc.get_referents(*queue))
        new = {o_id: o for o_id, o in referents if o_id not in marked and not isinstance(o, type)}
        queue = list(new.values())
        marked.update(new.keys())
    return total


def read_log_control_flow(log_basename: str) -> List[List[str]]:
    log = xes_importer.apply(os.path.join(ROOT_DIR, "event_logs", f"{log_basename}.xes.gz"))
    control_flow: List[List[str]] = []
    for trace in log:
        control_flow.append([e["concept:name"] for e in trace])
    return control_flow


def compute_embeddings_for_method(method: str, log_input: List[List[str]]):
    alphabet = get_alphabet(log_input)
    win_size = extract_window_size(method)

    if method == "Unit Distance":
        return {activity: np.eye(len(alphabet))[i] for i, activity in enumerate(alphabet)}

    if method.startswith("Bose 2009 Substitution Scores"):
        from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import (
            get_substitution_and_insertion_scores,
        )
        _, emb = get_substitution_and_insertion_scores(log_input, alphabet, win_size)
        return emb

    if method.startswith("De Koninck 2018 act2vec"):
        from distances.activity_distances.de_koninck_2018_act2vec.algorithm import (
            get_act2vec_distance_matrix,
        )
        sg = 0 if "CBOW" in method else 1
        _, emb = get_act2vec_distance_matrix(log_input, alphabet, sg, win_size)
        return emb

    if method.startswith("Our act2vec"):
        from distances.activity_distances.de_koninck_2018_act2vec.our_hyperparas import (
            get_act2vec_distance_matrix_our,
        )
        emb = get_act2vec_distance_matrix_our(log_input, alphabet, win_size)
        return emb

    if method.startswith("Activity-Activitiy Co Occurrence"):
        from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import (
            get_activity_activity_co_occurence_matrix,
        )
        from distances.activity_distances.pmi.pmi import (
            get_activity_activity_frequency_matrix_pmi,
        )
        bag_of_words = True if "Bag Of Words" in method else False
        _, emb, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(
            log_input, alphabet, win_size, bag_of_words
        )
        if "PPMI" in method:
            _, emb = get_activity_activity_frequency_matrix_pmi(emb, activity_freq_dict, activity_index, 1)
        elif "PMI" in method:
            _, emb = get_activity_activity_frequency_matrix_pmi(emb, activity_freq_dict, activity_index, 0)
        return emb

    if method.startswith("Activity-Context"):
        from distances.activity_distances.activity_context_frequency.activity_contex_frequency import (
            get_activity_context_frequency_matrix,
        )
        from distances.activity_distances.pmi.pmi import (
            get_activity_context_frequency_matrix_pmi,
        )
        if "Bag Of Words" in method and "N-Grams" not in method:
            bag_mode = 2
        elif "N-Grams" in method:
            bag_mode = 0
        else:
            bag_mode = 2
        _, emb, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(
            log_input, alphabet, win_size, bag_of_words=bag_mode
        )
        if "PPMI" in method:
            _, emb = get_activity_context_frequency_matrix_pmi(emb, activity_freq_dict, context_freq_dict, context_index, 1)
        elif "PMI" in method:
            _, emb = get_activity_context_frequency_matrix_pmi(emb, activity_freq_dict, context_freq_dict, context_index, 0)
        return emb

    if method.startswith("Chiorrini 2022 Embedding Process Structure"):
        from distances.activity_distances.chiorrini_2022_embedding_process_structure.embedding_process_structure import (
            get_embedding_process_structure_distance_matrix,
        )
        _, features = get_embedding_process_structure_distance_matrix(log_input, alphabet, False)
        return features

    if method.startswith("Gamallo Fernandez 2023 Context Based"):
        from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import (
            get_context_based_distance_matrix,
        )
        _, emb = get_context_based_distance_matrix(log_input, win_size)
        return emb

    raise ValueError(f"Unknown method: {method}")


def summarize_embedding_memory(emb) -> Dict[str, int]:
    # Basic counts
    num_items = len(emb) if isinstance(emb, dict) else 0
    # Try to infer consistent dimension
    dims = set()
    numpy_overhead = 0
    numpy_nbytes = 0
    values_overhead = 0
    for v in (emb.values() if isinstance(emb, dict) else []):
        if isinstance(v, np.ndarray):
            numpy_overhead += sys.getsizeof(v)
            numpy_nbytes += int(v.nbytes)
            if v.shape:
                dims.add(tuple(v.shape))
        else:
            values_overhead += sys.getsizeof(v)
            try:
                arr = np.asarray(v)
                if isinstance(arr, np.ndarray):
                    numpy_nbytes += int(arr.nbytes)
                    if arr.shape:
                        dims.add(tuple(arr.shape))
            except Exception:
                pass

    embedding_dim = -1
    if len(dims) == 1:
        shape = next(iter(dims))
        # Use last dimension as vector size, if 1D use that
        embedding_dim = int(shape[-1])

    shallow_dict_size = sys.getsizeof(emb) if isinstance(emb, dict) else 0
    keys_size = sum(sys.getsizeof(k) for k in (emb.keys() if isinstance(emb, dict) else []))

    deep_graph_size = get_obj_size(emb)
    # Deep graph does not include numpy data buffers, so add them explicitly
    estimated_total = deep_graph_size + numpy_nbytes

    return {
        "num_activities": num_items,
        "embedding_dim": embedding_dim,
        "shallow_dict_size_bytes": shallow_dict_size,
        "keys_size_bytes": keys_size,
        "values_overhead_bytes": values_overhead,
        "numpy_arrays_overhead_bytes": numpy_overhead,
        "numpy_arrays_nbytes": numpy_nbytes,
        "deep_object_graph_size_bytes": deep_graph_size,
        "estimated_total_bytes": estimated_total,
    }


def default_methods(include_gamallo: bool = False) -> List[str]:
    base = [
        "Unit Distance",
        "Bose 2009 Substitution Scores",
        "De Koninck 2018 act2vec CBOW",
        "De Koninck 2018 act2vec skip-gram",
        "Activity-Activitiy Co Occurrence Bag Of Words",
        "Activity-Context Bag Of Words",
        "Chiorrini 2022 Embedding Process Structure",
    ]
    if include_gamallo:
        base.append("Gamallo Fernandez 2023 Context Based")
    return base


def expand_methods_with_windows(methods: List[str], window_sizes: List[int]) -> List[str]:
    expanded: List[str] = []
    for m in methods:
        if m.startswith((
            "Bose", "De Koninck", "Activity-Activitiy", "Activity-Context", "Our", "Gamallo"
        )):
            for w in window_sizes:
                expanded.append(f"{m} w_{w}")
        else:
            expanded.append(m)
    return expanded


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_csv_header_if_needed(csv_path: str) -> None:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        ensure_parent_dir(csv_path)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "log",
                "method",
                "num_activities",
                "embedding_dim",
                "shallow_dict_size_mb",
                "keys_size_mb",
                "values_overhead_mb",
                "numpy_arrays_overhead_mb",
                "numpy_arrays_nbytes_mb",
                "deep_object_graph_size_mb",
                "estimated_total_mb",
            ])


def append_result(csv_path: str, log_name: str, method: str, m: Dict[str, int]) -> None:
    mb = 1024 ** 2
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            log_name,
            method,
            int(m["num_activities"]),
            int(m["embedding_dim"]),
            round(m["shallow_dict_size_bytes"] / mb, 3),
            round(m["keys_size_bytes"] / mb, 3),
            round(m["values_overhead_bytes"] / mb, 3),
            round(m["numpy_arrays_overhead_bytes"] / mb, 3),
            round(m["numpy_arrays_nbytes"] / mb, 3),
            round(m["deep_object_graph_size_bytes"] / mb, 3),
            round(m["estimated_total_bytes"] / mb, 3),
        ])


def main() -> None:
    if USE_CLI:
        parser = argparse.ArgumentParser(description="Compute and report RAM usage of embedding objects")
        parser.add_argument("--logs", type=str, default="Sepsis",
                            help="Comma-separated list of log base names in event_logs (without extension)")
        parser.add_argument("--methods", type=str, default="",
                            help="Comma-separated list of methods; if empty, use defaults")
        parser.add_argument("--include-gamallo", action="store_true",
                            help="Include Gamallo 2023 method in defaults")
        parser.add_argument("--window-sizes", type=str, default="3,5,9",
                            help="Comma-separated window sizes for applicable methods")
        parser.add_argument("--output", type=str, default=os.path.join(
            ROOT_DIR, "results", f"embedding_memory_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"))

        args = parser.parse_args()
        logs = [s.strip() for s in args.logs.split(',') if s.strip()]
        if args.methods.strip():
            methods_raw = [s.strip() for s in args.methods.split(',') if s.strip()]
        else:
            methods_raw = default_methods(include_gamallo=args.include_gamallo)
        window_sizes = [int(s) for s in args.window_sizes.split(',') if s.strip()]
        output_path = args.output
    else:
        logs = USER_CONFIG["logs"]
        methods_raw = USER_CONFIG["methods"] if USER_CONFIG["methods"] else default_methods(
            include_gamallo=USER_CONFIG["include_gamallo"]
        )
        window_sizes = USER_CONFIG["window_sizes"]
        output_path = USER_CONFIG["output"]

    methods = expand_methods_with_windows(methods_raw, window_sizes)

    write_csv_header_if_needed(output_path)

    for log_name in logs:
        cf = read_log_control_flow(log_name)
        for method in methods:
            print(f"[{log_name}] Method '{method}' - computing embeddings ...", flush=True)
            try:
                emb = compute_embeddings_for_method(method, cf)
                metrics = summarize_embedding_memory(emb)
                append_result(output_path, log_name, method, metrics)
                print(
                    f"  n={metrics['num_activities']}, dim={metrics['embedding_dim']}, "
                    f"estimated_total={metrics['estimated_total_bytes']/ (1024**2):.2f} MB",
                    flush=True,
                )
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()



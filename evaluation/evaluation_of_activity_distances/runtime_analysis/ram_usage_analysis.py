import argparse
import csv
import os
import threading
import time
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np
import psutil
from pm4py.objects.log.importer.xes import importer as xes_importer

# Ensure project root is importable when running this file directly by path
import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from definitions import ROOT_DIR

# =====================
# In-file configuration for IDE runs
# Flip USE_CLI to True if you want to control via command line again.
USE_CLI: bool = False
USER_CONFIG = {
    "logs": ["Sepsis"],
    # Example: ["Unit Distance", "De Koninck 2018 act2vec CBOW w_3"], or [] to use defaults
    "methods": ["Unit Distance"],
    "include_gamallo": False,
    "window_sizes": [3, 5, 9],
    "repetitions": 1,
    "sample_interval_ms": 200,
    "output": os.path.join(
        ROOT_DIR, "results", f"ram_usage_results_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ),
}
# =====================
import re
import gc


def extract_window_size(text: str) -> int:
    match = re.search(r"w_(\d+)", text)
    return int(match.group(1)) if match else 3


def get_alphabet(log: List[List[str]]) -> List[str]:
    unique_activities = set()
    for trace in log:
        for activity in trace:
            unique_activities.add(activity)
    return sorted(list(unique_activities))


def get_obj_size(obj) -> int:
    if obj is None:
        return 0
    marked = {id(obj)}
    obj_q = [obj]
    total_size = 0
    while obj_q:
        total_size += sum(map(sys.getsizeof, obj_q))
        all_referents = ((id(o), o) for o in gc.get_referents(*obj_q))
        new_referents = {o_id: o for o_id, o in all_referents if o_id not in marked and not isinstance(o, type)}
        obj_q = list(new_referents.values())
        marked.update(new_referents.keys())
    return total_size


def read_log_control_flow(log_basename: str) -> List[List[str]]:
    # Try .xes.gz first, then .xes
    gz_path = os.path.join(ROOT_DIR, "event_logs", f"{log_basename}.xes.gz")
    xes_path = os.path.join(ROOT_DIR, "event_logs", f"{log_basename}.xes")
    if os.path.exists(gz_path):
        log_path = gz_path
    elif os.path.exists(xes_path):
        log_path = xes_path
    else:
        raise FileNotFoundError(f"Event log not found. Tried: {gz_path} and {xes_path}")

    log = xes_importer.apply(log_path)
    # Convert to control-flow sequences (list of list of activity names)
    log_control_flow: List[List[str]] = []
    for trace in log:
        activities = [evt["concept:name"] for evt in trace]
        log_control_flow.append(activities)
    return log_control_flow


def get_total_rss_bytes(process: Optional[psutil.Process] = None) -> int:
    if process is None:
        process = psutil.Process(os.getpid())
    rss = 0
    try:
        rss += process.memory_info().rss
    except psutil.Error:
        pass
    # Include child processes if any
    try:
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                continue
    except psutil.Error:
        pass
    return rss


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self.peak_rss_bytes = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            current_rss = get_total_rss_bytes()
            if current_rss > self.peak_rss_bytes:
                self.peak_rss_bytes = current_rss
            time.sleep(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()


def compute_embeddings_for_method(method: str, log_input: List[List[str]]) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for a given method, returning a mapping from activity to embedding vector.
    The method string can include window size suffix like "... w_3"; if absent, default is 3 where applicable.
    """
    alphabet = sorted(set(token for trace in log_input for token in trace))
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
        # Returns (distances, features)
        _, features = get_embedding_process_structure_distance_matrix(log_input, alphabet, False)
        return features

    if method.startswith("Gamallo Fernandez 2023 Context Based"):
        from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import (
            get_context_based_distance_matrix,
        )
        # Returns (distances, embeddings)
        _, emb = get_context_based_distance_matrix(log_input, win_size)
        return emb

    raise ValueError(f"Unknown method: {method}")


def measure_ram_for_method(method: str, log_cf: List[List[str]], sample_interval_s: float) -> Dict[str, float]:
    process = psutil.Process(os.getpid())
    baseline_rss = get_total_rss_bytes(process)
    sampler = MemorySampler(interval_seconds=sample_interval_s)
    embeddings: Optional[Dict[str, np.ndarray]] = None
    error_message: Optional[str] = None

    start_time = time.time()
    sampler.start()
    try:
        embeddings = compute_embeddings_for_method(method, log_cf)
    except Exception as e:  # noqa: BLE001
        error_message = str(e)
    finally:
        sampler.stop()
    duration_s = time.time() - start_time

    end_rss = get_total_rss_bytes(process)
    peak_rss = max(sampler.peak_rss_bytes, baseline_rss, end_rss)
    delta_peak = max(0, peak_rss - baseline_rss)

    embedding_size_bytes = get_obj_size(embeddings) if embeddings is not None else 0
    num_activities = len(embeddings) if embeddings is not None else 0

    return {
        "baseline_rss_bytes": float(baseline_rss),
        "peak_rss_bytes": float(peak_rss),
        "delta_peak_bytes": float(delta_peak),
        "end_rss_bytes": float(end_rss),
        "duration_seconds": float(duration_s),
        "embedding_object_size_bytes": float(embedding_size_bytes),
        "num_activities": float(num_activities),
        "error": 1.0 if error_message else 0.0,
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
    # Gamallo is heavy; include only on request
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
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def write_csv_header_if_needed(csv_path: str) -> None:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        ensure_parent_dir(csv_path)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "log",
                "method",
                "baseline_rss_mb",
                "peak_rss_mb",
                "delta_peak_mb",
                "end_rss_mb",
                "duration_seconds",
                "embedding_object_size_mb",
                "num_activities",
                "error",
            ])


def append_result(csv_path: str, log_name: str, method: str, metrics: Dict[str, float]) -> None:
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            log_name,
            method,
            round(metrics["baseline_rss_bytes"] / (1024**2), 3),
            round(metrics["peak_rss_bytes"] / (1024**2), 3),
            round(metrics["delta_peak_bytes"] / (1024**2), 3),
            round(metrics["end_rss_bytes"] / (1024**2), 3),
            round(metrics["duration_seconds"], 4),
            round(metrics["embedding_object_size_bytes"] / (1024**2), 3),
            int(metrics["num_activities"]),
            int(metrics["error"]),
        ])


def main() -> None:
    if USE_CLI:
        parser = argparse.ArgumentParser(description="Measure RAM usage during embedding computation")
        parser.add_argument(
            "--logs",
            type=str,
            default="Sepsis",
            help="Comma-separated list of log base names found in ROOT_DIR/event_logs (without extension)",
        )
        parser.add_argument(
            "--methods",
            type=str,
            default="",
            help="Comma-separated list of methods; if empty, use a sensible default set",
        )
        parser.add_argument(
            "--include-gamallo",
            action="store_true",
            help="Include Gamallo 2023 context-based embeddings in defaults",
        )
        parser.add_argument(
            "--window-sizes",
            type=str,
            default="3,5,9",
            help="Comma-separated window sizes applied to applicable methods",
        )
        parser.add_argument(
            "--repetitions",
            type=int,
            default=1,
            help="How many repetitions per method to average/observe",
        )
        parser.add_argument(
            "--sample-interval-ms",
            type=int,
            default=200,
            help="Sampling interval for memory in milliseconds",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=os.path.join(
                ROOT_DIR,
                "results",
                f"ram_usage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            ),
            help="Path to output CSV file",
        )

        args = parser.parse_args()
        log_names = [s.strip() for s in args.logs.split(",") if s.strip()]
        if args.methods.strip():
            methods_raw = [s.strip() for s in args.methods.split(",") if s.strip()]
        else:
            methods_raw = default_methods(include_gamallo=args.include_gamallo)
        window_sizes = [int(s) for s in args.window_sizes.split(",") if s.strip()]
        repetitions = args.repetitions
        sample_interval_s = max(0.01, args.sample_interval_ms / 1000.0)
        output_path = args.output
    else:
        log_names = USER_CONFIG["logs"]
        methods_raw = USER_CONFIG["methods"] if USER_CONFIG["methods"] else default_methods(
            include_gamallo=USER_CONFIG["include_gamallo"]
        )
        window_sizes = USER_CONFIG["window_sizes"]
        repetitions = USER_CONFIG["repetitions"]
        sample_interval_s = max(0.01, USER_CONFIG["sample_interval_ms"] / 1000.0)
        output_path = USER_CONFIG["output"]

    methods = expand_methods_with_windows(methods_raw, window_sizes)

    write_csv_header_if_needed(output_path)

    for log_name in log_names:
        log_cf = read_log_control_flow(log_name)
        for method in methods:
            for rep in range(repetitions):
                print(f"[{log_name}] Method '{method}' (rep {rep + 1}/{repetitions}) ...", flush=True)
                metrics = measure_ram_for_method(method, log_cf, sample_interval_s)
                append_result(output_path, log_name, method, metrics)
                print(
                    f"  peak={metrics['peak_rss_bytes'] / (1024**2):.2f} MB, "
                    f"delta={metrics['delta_peak_bytes'] / (1024**2):.2f} MB, "
                    f"size={metrics['embedding_object_size_bytes'] / (1024**2):.2f} MB, "
                    f"n={int(metrics['num_activities'])}, "
                    f"t={metrics['duration_seconds']:.2f}s, error={int(metrics['error'])}",
                    flush=True,
                )


if __name__ == "__main__":
    main()



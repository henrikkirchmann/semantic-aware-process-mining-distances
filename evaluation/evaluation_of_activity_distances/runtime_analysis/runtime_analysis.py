from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure repo root is importable when running this file directly by path (IDE / python -u ...)
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR

# Uncertain runtime evaluation imports (kept TensorFlow-free)
from uncertain_utils.uncertain_xes_reader import read_uncertain_xes
from evaluation.data_util.uncertain_evaluation_helpers import add_window_size_evaluation
from evaluation.data_util.util_activity_distances_intrinsic_uncertain import apply_uncertainty_level
from evaluation.data_util.util_activity_distances_uncertain import (
    UNCERTAIN_COUNT_BASED_METHODS,
    UNCERTAIN_NEURAL_METHODS,
    get_uncertain_activity_distance_matrix,
)


def extract_window_size(s: str) -> int:
    """
    Extract window size from method strings like "... w_3".
    Local copy to avoid importing deterministic utilities (which pull in heavy deps in some environments).
    """
    m = re.search(r"w_(\d+)", s)
    return int(m.group(1)) if m else 3


def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        pass


def evaluate_runtime(activity_distance_functions, log_list, number_of_repetitions, *, verbose: bool = True):
    """
    Deterministic runtime evaluation (legacy).

    Note: heavy imports are done lazily inside this function so that uncertain mode can run without
    optional dependencies like TensorFlow.
    """
    # Deterministic runtime evaluation imports (legacy; can pull optional deps)
    from pm4py.objects.log.importer.xes import importer as xes_importer
    from evaluation.data_util.util_activity_distances import (
        get_alphabet,
        get_unit_cost_activity_distance_matrix,
    )
    from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
    from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import (
        get_substitution_and_insertion_scores,
    )
    from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
    from distances.activity_distances.chiorrini_2022_embedding_process_structure.embedding_process_structure import (
        get_embedding_process_structure_distance_matrix,
    )
    from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import (
        get_context_based_distance_matrix,
    )
    from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import (
        get_activity_activity_co_occurence_matrix,
    )
    from distances.activity_distances.activity_context_frequency.activity_contex_frequency import (
        get_activity_context_frequency_matrix,
    )
    from distances.activity_distances.pmi.pmi import (
        get_activity_activity_frequency_matrix_pmi,
        get_activity_context_frequency_matrix_pmi,
    )

    results = []

    for log_idx, log_name in enumerate(log_list, start=1):
        # Import the event log
        log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes.gz')
        log_control_flow_perspective = get_log_control_flow_perspective(log)
        alphabet = get_alphabet(log_control_flow_perspective)
        if verbose:
            n_traces = len(log_control_flow_perspective)
            n_events = sum(len(t) for t in log_control_flow_perspective)
            _safe_print(
                f"[runtime:det] log {log_idx}/{len(log_list)}: {log_name} "
                f"(traces={n_traces}, events={n_events}, |A|={len(alphabet)})"
            )

        for m_idx, activity_distance_function in enumerate(activity_distance_functions, start=1):
            runtimes = []
            for _ in range(number_of_repetitions):
                window_size = extract_window_size(activity_distance_function)
                if verbose:
                    _safe_print(
                        f"[runtime:det]  method {m_idx}/{len(activity_distance_functions)}: "
                        f"{activity_distance_function} (w={window_size})"
                    )

                if activity_distance_function.startswith("Bose 2009 Substitution Scores"):
                    start_time = time.time()
                    activity_distance_matrix, embedding = get_substitution_and_insertion_scores(
                        log_control_flow_perspective,
                        alphabet, window_size)
                    runtimes.append(time.time() - start_time)
                elif activity_distance_function.startswith("De Koninck 2018 act2vec"):
                    if "CBOW" in activity_distance_function:
                        sg = 0
                    else:
                        sg = 1
                    start_time = time.time()
                    act2vec_distance_matrix, embedding = get_act2vec_distance_matrix(
                        log_control_flow_perspective,
                        alphabet, sg, window_size)
                    runtimes.append(time.time() - start_time)

                elif activity_distance_function.startswith("Unit Distance"):
                    start_time = time.time()

                    unit_distance_matrix, emb = get_unit_cost_activity_distance_matrix(
                        log_control_flow_perspective,
                        alphabet)
                    runtimes.append(time.time() - start_time)

                elif "Chiorrini 2022 Embedding Process Structure" in activity_distance_function:
                    # to make the authors implmenetation work we have to some I/O operations, with pnml files,
                    # thus we measure the time without them, and implemented the time measure inside the called function
                    # 1 for time measurement with pm discovery (inductive miner)
                    # 2 for time measurement without pm discovery
                    if "Discovery" in activity_distance_function:
                        runtimes.append(
                            get_embedding_process_structure_distance_matrix(log_control_flow_perspective, alphabet, 1))
                        print("w discovery")
                    else:
                        runtimes.append(
                            get_embedding_process_structure_distance_matrix(log_control_flow_perspective, alphabet, 2))
                        print("wo discovery")


                elif "Gamallo Fernandez 2023 Context Based" in activity_distance_function:
                    runtimes.append(get_context_based_distance_matrix(
                        log_control_flow_perspective, window_size, take_time=True))
                elif activity_distance_function.startswith("Activity-Activitiy Co Occurrence Bag Of Words"):
                    start_time = time.time()

                    activity_distance_matrix, embedding, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(
                        log_control_flow_perspective,
                        alphabet, window_size, True)
                    if "PPMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_activity_frequency_matrix_pmi(embedding,
                                                                                                         activity_freq_dict,
                                                                                                         activity_index,
                                                                                                         1)
                        runtimes.append(time.time() - start_time)

                    elif "PMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_activity_frequency_matrix_pmi(embedding,
                                                                                                         activity_freq_dict,
                                                                                                         activity_index,
                                                                                                         0)
                        runtimes.append(time.time() - start_time)

                    else:
                        runtimes.append(time.time() - start_time)


                elif activity_distance_function.startswith("Activity-Activitiy Co Occurrence N-Gram"):
                    start_time = time.time()

                    activity_distance_matrix, embedding, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(
                        log_control_flow_perspective,
                        alphabet, window_size, False)
                    if "PPMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_activity_frequency_matrix_pmi(embedding,
                                                                                                         activity_freq_dict,
                                                                                                         activity_index,
                                                                                                         1)
                        runtimes.append(time.time() - start_time)
                    elif "PMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_activity_frequency_matrix_pmi(embedding,
                                                                                                         activity_freq_dict,
                                                                                                         activity_index,
                                                                                                         0)
                        runtimes.append(time.time() - start_time)
                    else:
                        runtimes.append(time.time() - start_time)

                elif activity_distance_function.startswith("Activity-Context Bag Of Words"):
                    start_time = time.time()
                    activity_distance_matrix, embedding, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(
                        log_control_flow_perspective,
                        alphabet, window_size, 2)
                    if "PPMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        1)
                        runtimes.append(time.time() - start_time)

                    elif "PMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        0)
                        runtimes.append(time.time() - start_time)
                    else:
                        runtimes.append(time.time() - start_time)



                elif "Activity-Context as Bag of Words as N-Grams" in activity_distance_function:
                    start_time = time.time()

                    activity_distance_matrix, embedding, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(
                        log_control_flow_perspective,
                        alphabet, window_size, 1)
                    if "PPMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        1)
                        runtimes.append(time.time() - start_time)
                    elif "PMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        0)
                        runtimes.append(time.time() - start_time)
                    else:
                        runtimes.append(time.time() - start_time)
                elif "Activity-Context N-Grams" in activity_distance_function:
                    start_time = time.time()
                    activity_distance_matrix, embedding, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(
                        log_control_flow_perspective,
                        alphabet, window_size, 0)
                    if "PPMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        1)
                        runtimes.append(time.time() - start_time)

                    elif "PMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        0)
                        runtimes.append(time.time() - start_time)
                    else:
                        runtimes.append(time.time() - start_time)
                else:
                    raise ValueError("Unknown encoding method: " + activity_distance_function)

            # Calculate the average runtime
            avg_runtime = sum(runtimes) / len(runtimes)
            row = {
                "activity_function_name": activity_distance_function,
                "log": log_name,
                "average_duration": avg_runtime
            }
            _safe_print(str(row))
            results.append(row)

    return results


def evaluate_runtime_uncertain(
    activity_distance_functions,
    log_list,
    number_of_repetitions,
    *,
    na_label: str = "NA",
    # NOTE: For the runtime benchmark we match the intrinsic benchmark semantics:
    # apply top-u per event + renormalize *before* running the methods.
    uncertainty_level_u: int = 3,
    limit_traces: int | None = None,
    inner_progress_every: int = 200_000,
    verbose: bool = True,
):
    """
    Runtime evaluation for *uncertain* event logs.

    - Reads XES from `<ROOT_DIR>/uncertain_event_logs/<log_name>.xes` (or .xes.gz if present)
    - Uses `get_uncertain_activity_distance_matrix(...)` for the method runtime.
    """
    results = []

    for log_idx, log_name in enumerate(log_list, start=1):
        xes = Path(ROOT_DIR) / "uncertain_event_logs" / f"{log_name}.xes"
        xes_gz = Path(ROOT_DIR) / "uncertain_event_logs" / f"{log_name}.xes.gz"
        path = xes_gz if xes_gz.exists() else xes
        if not path.exists():
            raise FileNotFoundError(f"Uncertain log not found: {path}")

        log_u_raw = read_uncertain_xes(str(path), limit_traces=limit_traces)
        log_u = apply_uncertainty_level(log_u_raw, k=int(uncertainty_level_u), na_label=na_label)
        if verbose:
            n_traces = len(log_u.traces)
            n_events = sum(len(tr.events) for tr in log_u.traces)
            _safe_print(
                f"[runtime:unc] log {log_idx}/{len(log_list)}: {log_name} "
                f"(traces={n_traces}, events={n_events}, u={int(uncertainty_level_u)})"
            )

        for m_idx, activity_distance_function in enumerate(activity_distance_functions, start=1):
            runtimes = []
            window_size = extract_window_size(activity_distance_function)
            method_name = activity_distance_function.replace(f" w_{window_size}", "")

            if verbose:
                _safe_print(
                    f"[runtime:unc]  method {m_idx}/{len(activity_distance_functions)}: "
                    f"{method_name} (w={window_size})"
                )

            for rep in range(1, number_of_repetitions + 1):
                window_size = extract_window_size(activity_distance_function)
                method_name = activity_distance_function.replace(f" w_{window_size}", "")

                start_time = time.time()
                if verbose:
                    _safe_print(f"[runtime:unc]    rep {rep}/{number_of_repetitions} start ...")
                _dist, _dbg = get_uncertain_activity_distance_matrix(
                    log_u,
                    method_name=method_name,
                    window_size=window_size,
                    top_k=None,
                    min_prob=0.0,
                    na_label=na_label,
                    progress=_safe_print if verbose else None,
                    progress_every_realizations=int(inner_progress_every),
                )
                dt = time.time() - start_time
                runtimes.append(dt)
                if verbose:
                    _safe_print(f"[runtime:unc]    rep {rep}/{number_of_repetitions} done in {dt:.2f}s")

            avg_runtime = sum(runtimes) / len(runtimes)
            row = {
                "activity_function_name": activity_distance_function,
                "log": log_name,
                "average_duration": avg_runtime,
                "mode": "uncertain",
                "u": int(uncertainty_level_u),
                "limit_traces": limit_traces,
            }
            _safe_print(str(row))
            results.append(row)

    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["deterministic", "uncertain"],
        default="uncertain",
        help="Which runtime evaluation to run (default: uncertain).",
    )
    ap.add_argument("--repetitions", type=int, default=3, help="Number of repetitions per method/log (default: 3).")
    ap.add_argument("--limit-traces", type=int, default=None, help="Optional: only read the first N traces.")
    ap.add_argument("--na-label", type=str, default="NA", help="NA label for uncertain logs (default: NA).")
    ap.add_argument(
        "--u",
        type=int,
        default=3,
        help="Uncertainty level: keep top-u labels per event and renormalize before running methods (default: 3).",
    )
    ap.add_argument(
        "--inner-progress-every",
        type=int,
        default=200_000,
        help="For uncertain methods: print inner progress every N window realizations (default: 200000).",
    )
    ap.add_argument("--quiet", action="store_true", help="Disable progress printing.")
    args = ap.parse_args()

    number_of_repetitions = int(args.repetitions)
    verbose = not bool(args.quiet)

    if args.mode == "uncertain":
        # Uncertain methods (12 count-based + 2 neural), evaluated for window sizes 3/5/9
        window_size_list = [3, 5, 9]
        activity_distance_functions = add_window_size_evaluation(
            list(UNCERTAIN_COUNT_BASED_METHODS) + list(UNCERTAIN_NEURAL_METHODS),
            window_size_list,
        )

        # Default logs requested by the user:
        # - ResNet-18 (RGB, dev3)
        # - ST-GCN-64 (Pose, dev3)
        log_list = [
            "ikea_asm__frame_based__resnet18__pretrained__rgb__dev3__xes_uncertain_pred_merged",
            "ikea_asm__pose_based__ST_GCN_64__pretrained__pose__dev3__xes_uncertain_pred_merged",
        ]

        print(f"Evaluating UNCERTAIN runtimes with {number_of_repetitions} repetitions...")
        runtime_results = evaluate_runtime_uncertain(
            activity_distance_functions,
            log_list,
            number_of_repetitions,
            na_label=str(args.na_label),
            uncertainty_level_u=int(args.u),
            limit_traces=(int(args.limit_traces) if args.limit_traces is not None else None),
            inner_progress_every=int(args.inner_progress_every),
            verbose=verbose,
        )

        csv_filename = f"runtime_results_uncertain_{number_of_repetitions}_repetitions.csv"
    else:
        # Deterministic (legacy): keep old behavior
        activity_distance_functions = []
        activity_distance_functions.append("Unit Distance")
        activity_distance_functions.append("Bose 2009 Substitution Scores")
        activity_distance_functions.append("De Koninck 2018 act2vec CBOW")
        activity_distance_functions.append("Chiorrini 2022 Embedding Process Structure")
        activity_distance_functions.append("Chiorrini 2022 Embedding Process Structure Discovery")
        activity_distance_functions.append("De Koninck 2018 act2vec skip-gram")
        activity_distance_functions.append("Activity-Activitiy Co Occurrence Bag Of Words")
        activity_distance_functions.append("Activity-Activitiy Co Occurrence N-Gram")
        activity_distance_functions.append("Activity-Activitiy Co Occurrence Bag Of Words PMI")
        activity_distance_functions.append("Activity-Activitiy Co Occurrence N-Gram PMI")
        activity_distance_functions.append("Activity-Activitiy Co Occurrence Bag Of Words PPMI")
        activity_distance_functions.append("Activity-Activitiy Co Occurrence N-Gram PPMI")
        activity_distance_functions.append("Activity-Context Bag Of Words")
        activity_distance_functions.append("Activity-Context N-Grams")
        activity_distance_functions.append("Activity-Context Bag Of Words PMI")
        activity_distance_functions.append("Activity-Context N-Grams PMI")
        activity_distance_functions.append("Activity-Context Bag Of Words PPMI")
        activity_distance_functions.append("Activity-Context N-Grams PPMI")

        from evaluation.data_util.util_activity_distances_intrinsic import add_window_size_evaluation as _add_ws_det
        window_size_list = [3, 5, 9]
        activity_distance_functions = _add_ws_det(activity_distance_functions, window_size_list)
        activity_distance_functions.append("Gamallo Fernandez 2023 Context Based w_3")
        log_list = ["Sepsis"]

        print(f"Evaluating deterministic runtimes with {number_of_repetitions} repetitions...")
        runtime_results = evaluate_runtime(activity_distance_functions, log_list, number_of_repetitions, verbose=verbose)
        csv_filename = f"runtime_results_{number_of_repetitions}_repetitions_gamallo_19.csv"

    out_path = Path(ROOT_DIR) / "results" / csv_filename
    df = pd.DataFrame(runtime_results)
    df.to_csv(str(out_path), index=False)
    print(f"Results saved to {out_path}")

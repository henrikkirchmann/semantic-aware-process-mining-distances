
"""Tools to align activities between two event logs using embedding alignment."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pm4py

from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import (
    get_activity_activity_co_occurence_matrix,
)
from distances.activity_distances.activity_context_frequency.activity_contex_frequency import (
    get_activity_context_frequency_matrix,
)
from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import (
    get_substitution_and_insertion_scores,
)
from distances.activity_distances.chiorrini_2022_embedding_process_structure.embedding_process_structure import (
    get_embedding_process_structure_distance_matrix,
)
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import (
    get_act2vec_distance_matrix,
)
from distances.activity_distances.de_koninck_2018_act2vec.our_hyperparas import (
    get_act2vec_distance_matrix_our,
)
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import (
    get_context_based_distance_matrix,
)
from distances.activity_distances.pmi.pmi import (
    get_activity_activity_frequency_matrix_pmi,
    get_activity_context_frequency_matrix_pmi,
)
from evaluation.data_util.util_activity_distances_intrinsic import (
    add_window_size_evaluation,
)


LOGGER = logging.getLogger(__name__)


BASE_EMBEDDING_METHODS = [
    "one_hot",
    "Uniform Zero Embedding",
    "Random Uniform Embedding",
    "Unit Distance",
    "Bose 2009 Substitution Scores",
    "De Koninck 2018 act2vec CBOW",
    "De Koninck 2018 act2vec skip-gram",
    "Our act2vec",
    "Activity-Activitiy Co Occurrence Bag Of Words",
    "Activity-Activitiy Co Occurrence N-Gram",
    "Activity-Activitiy Co Occurrence Bag Of Words PMI",
    "Activity-Activitiy Co Occurrence N-Gram PMI",
    "Activity-Activitiy Co Occurrence Bag Of Words PPMI",
    "Activity-Activitiy Co Occurrence N-Gram PPMI",
    "Activity-Context Bag Of Words",
    "Activity-Context N-Grams",
    "Activity-Context Bag Of Words PMI",
    "Activity-Context N-Grams PMI",
    "Activity-Context Bag Of Words PPMI",
    "Activity-Context N-Grams PPMI",
    "Chiorrini 2022 Embedding Process Structure",
]

WINDOW_SIZES = [3, 5, 9]

SUPPORTED_EMBEDDING_METHODS = add_window_size_evaluation(
    BASE_EMBEDDING_METHODS, WINDOW_SIZES
)
SUPPORTED_EMBEDDING_METHODS.append("Gamallo Fernandez 2023 Context Based w_3")
SUPPORTED_EMBEDDING_METHODS = sorted(set(SUPPORTED_EMBEDDING_METHODS))


def read_event_log(path: Path) -> Sequence[Sequence[str]]:
    """Read an event log from ``path`` and return the control-flow perspective."""

    LOGGER.info("Reading event log from %s", path)
    log = pm4py.read_xes(str(path))
    control_flow = [[event["concept:name"] for event in trace] for trace in log]
    if not control_flow:
        raise ValueError(f"No traces found in event log: {path}")
    return control_flow


def compute_union_alphabet(*logs: Sequence[Sequence[str]]) -> Sequence[str]:
    """Return the sorted union of activities from ``logs``."""

    activities = {activity for log in logs for trace in log for activity in trace}
    if not activities:
        raise ValueError("No activities found across the provided logs.")
    return sorted(activities)


def extract_window_size(method: str) -> int | None:
    """Extract the window size encoded in ``method`` (``w_<size>``) if present."""

    match = re.search(r"w_(\d+)", method)
    return int(match.group(1)) if match else None


def _normalize_log(log_control_flow: Sequence[Sequence[str]]) -> Sequence[Sequence[str]]:
    """Return a version of the log suitable for embedding utilities."""

    normalized = []
    for trace in log_control_flow:
        cleaned_trace = [str(activity) for activity in trace if activity is not None]
        if cleaned_trace:
            normalized.append(cleaned_trace)
    return normalized


def compute_method_embeddings(
    log_control_flow: Sequence[Sequence[str]],
    alphabet: Sequence[str],
    method: str,
    window_size: int | None,
    rng: np.random.Generator | None = None,
) -> Dict[str, np.ndarray]:
    """Compute embeddings for ``log_control_flow`` using ``method``.

    The implementation mirrors the embedding selection logic used in the
    ``next_activity_prediction_everman`` pipeline so that experiments can reuse
    the exact same representations.
    """

    if rng is None:
        rng = np.random.default_rng()

    normalized_log = _normalize_log(log_control_flow)
    if not normalized_log:
        raise ValueError("The event log is empty after filtering missing activities.")

    full_alphabet = sorted({str(activity) for activity in alphabet} |
                           {event for trace in normalized_log for event in trace})
    if not full_alphabet:
        raise ValueError("Unable to determine an activity alphabet for embeddings.")

    resolved_window = window_size or extract_window_size(method) or WINDOW_SIZES[0]
    LOGGER.debug(
        "Computing embeddings using method=%s, window_size=%d, alphabet_size=%d",
        method,
        resolved_window,
        len(full_alphabet),
    )

    method = method.strip()
    embeddings: MutableMapping[str, np.ndarray]

    if method == "one_hot":
        identity = np.eye(len(full_alphabet), dtype=np.float32)
        embeddings = {activity: identity[i] for i, activity in enumerate(full_alphabet)}
    elif method == "Uniform Zero Embedding":
        dim = len(full_alphabet)
        embeddings = {
            activity: np.zeros(dim, dtype=np.float32) for activity in full_alphabet
        }
    elif method == "Random Uniform Embedding":
        dim = len(full_alphabet)
        embeddings = {
            activity: rng.uniform(-10, 10, size=(dim,)).astype(np.float32)
            for activity in full_alphabet
        }
    elif method.startswith("Unit Distance"):
        identity = np.eye(len(full_alphabet), dtype=np.float32)
        embeddings = {activity: identity[i] for i, activity in enumerate(full_alphabet)}
    elif method.startswith("Bose 2009 Substitution Scores"):
        _, raw_embeddings = get_substitution_and_insertion_scores(
            normalized_log, list(full_alphabet), resolved_window
        )
        embeddings = {
            activity: np.asarray(vector, dtype=np.float32)
            for activity, vector in raw_embeddings.items()
        }
    elif method.startswith("De Koninck 2018 act2vec"):
        sg = 0 if "CBOW" in method else 1
        _, raw_embeddings = get_act2vec_distance_matrix(
            normalized_log, list(full_alphabet), sg, resolved_window
        )
        embeddings = {
            activity: np.asarray(vector, dtype=np.float32)
            for activity, vector in raw_embeddings.items()
        }
    elif method.startswith("Our act2vec"):
        raw_embeddings = get_act2vec_distance_matrix_our(
            normalized_log, list(full_alphabet), resolved_window
        )
        embeddings = {
            activity: np.asarray(vector, dtype=np.float32)
            for activity, vector in raw_embeddings.items()
        }
    elif method.startswith("Activity-Activitiy Co Occurrence"):
        bag = "Bag Of Words" in method
        _, raw_embeddings, activity_freq_dict, activity_index = (
            get_activity_activity_co_occurence_matrix(
                normalized_log,
                list(full_alphabet),
                resolved_window,
                bag_of_words=bag,
            )
        )
        if "PPMI" in method:
            _, raw_embeddings = get_activity_activity_frequency_matrix_pmi(
                raw_embeddings, activity_freq_dict, activity_index, 1
            )
        elif "PMI" in method:
            _, raw_embeddings = get_activity_activity_frequency_matrix_pmi(
                raw_embeddings, activity_freq_dict, activity_index, 0
            )
        embeddings = {
            activity: np.asarray(vector, dtype=np.float32)
            for activity, vector in raw_embeddings.items()
        }
    elif method.startswith("Activity-Context"):
        if "Bag Of Words" in method and "N-Grams" not in method:
            bag_mode = 2
        elif "N-Grams" in method:
            bag_mode = 0
        else:
            bag_mode = 2
        (
            _,
            raw_embeddings,
            activity_freq_dict,
            context_freq_dict,
            context_index,
        ) = get_activity_context_frequency_matrix(
            normalized_log,
            list(full_alphabet),
            resolved_window,
            bag_of_words=bag_mode,
        )
        if "PPMI" in method:
            _, raw_embeddings = get_activity_context_frequency_matrix_pmi(
                raw_embeddings, activity_freq_dict, context_freq_dict, context_index, 1
            )
        elif "PMI" in method:
            _, raw_embeddings = get_activity_context_frequency_matrix_pmi(
                raw_embeddings, activity_freq_dict, context_freq_dict, context_index, 0
            )
        embeddings = {
            activity: np.asarray(vector, dtype=np.float32)
            for activity, vector in raw_embeddings.items()
        }
    elif method.startswith("Chiorrini 2022 Embedding Process Structure"):
        _, raw_embeddings = get_embedding_process_structure_distance_matrix(
            normalized_log, list(full_alphabet), False
        )
        embeddings = {
            activity: np.asarray(vector, dtype=np.float32)
            for activity, vector in raw_embeddings.items()
        }
    elif method.startswith("Gamallo Fernandez 2023 Context Based"):
        _, raw_embeddings = get_context_based_distance_matrix(
            normalized_log, resolved_window
        )
        embeddings = {
            activity: np.asarray(vector, dtype=np.float32)
            for activity, vector in raw_embeddings.items()
        }
    else:
        raise ValueError(f"Unknown embedding method: {method}")

    if not embeddings:
        raise ValueError(
            f"Embedding method '{method}' produced no vectors. Check the input log."
        )

    dimension = len(next(iter(embeddings.values())))
    for activity in full_alphabet:
        embeddings.setdefault(activity, np.zeros(dimension, dtype=np.float32))

    return dict(embeddings)


def compute_embeddings(
    log_control_flow: Sequence[Sequence[str]],
    alphabet: Sequence[str],
    method: str,
    window_size: int | None,
    rng: np.random.Generator | None = None,
) -> Tuple[Dict[str, np.ndarray], Mapping[str, int]]:
    """Compute activity embeddings for ``log_control_flow`` using ``method``."""

    embeddings = compute_method_embeddings(log_control_flow, alphabet, method, window_size, rng)
    frequency = Counter(
        activity for trace in log_control_flow for activity in trace if activity is not None
    )
    return embeddings, frequency


def write_embeddings(embeddings: Mapping[str, np.ndarray], path: Path) -> Sequence[str]:
    """Write embeddings to ``path`` in the text format expected by MUSE."""

    activities = sorted(embeddings.keys())
    if not activities:
        raise ValueError("No embeddings available to write.")
    dim = next(iter(embeddings.values())).shape[0]
    with path.open("w", encoding="utf-8") as file:
        file.write(f"{len(activities)} {dim}\n")
        for activity in activities:
            vector = embeddings[activity]
            if vector.shape[0] != dim:
                raise ValueError("Embedding dimension mismatch detected.")
            vector_str = " ".join(f"{value:.6f}" for value in vector)
            file.write(f"{activity} {vector_str}\n")
    LOGGER.info("Wrote %d embeddings (dim=%d) to %s", len(activities), dim, path)
    return activities


def run_muse_unsupervised(
    script_path: Path,
    src_emb: Path,
    tgt_emb: Path,
    src_lang: str,
    tgt_lang: str,
    extra_args: Sequence[str],
) -> None:
    """Execute the MUSE ``unsupervised.py`` script with the prepared embeddings."""

    command = [
        "python",
        os.fspath(script_path),
        "--src_lang",
        src_lang,
        "--tgt_lang",
        tgt_lang,
        "--src_emb",
        os.fspath(src_emb),
        "--tgt_emb",
        os.fspath(tgt_emb),
    ]
    command.extend(extra_args)

    LOGGER.info("Running MUSE unsupervised alignment: %s", " ".join(command))
    subprocess.run(command, check=True)


def prepare_metadata(
    output_dir: Path,
    src_label: str,
    tgt_label: str,
    alphabet: Sequence[str],
    src_freq: Mapping[str, int],
    tgt_freq: Mapping[str, int],
    src_embedding_file: Path,
    tgt_embedding_file: Path,
    embedding_method: str,
    window_size: int | None,
    random_seed: int,
) -> None:
    """Persist auxiliary information about the alignment inputs."""

    metadata = {
        "alphabet": list(alphabet),
        "embedding_method": embedding_method,
        "window_size": window_size,
        "random_seed": random_seed,
        "source": {
            "label": src_label,
            "embedding_file": os.fspath(src_embedding_file),
            "activity_frequencies": dict(src_freq),
        },
        "target": {
            "label": tgt_label,
            "embedding_file": os.fspath(tgt_embedding_file),
            "activity_frequencies": dict(tgt_freq),
        },
    }
    metadata_path = output_dir / "alignment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Wrote metadata to %s", metadata_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute activity embeddings for two event logs and optionally run "
            "MUSE's unsupervised alignment to infer activity correspondences."
        )
    )
    parser.add_argument("src_log", type=Path, help="Path to the source event log (.xes or .xes.gz)")
    parser.add_argument("tgt_log", type=Path, help="Path to the target event log (.xes or .xes.gz)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("alignment_output"),
        help="Directory where embeddings and metadata will be written.",
    )
    parser.add_argument(
        "--src-label",
        default="src",
        help="Label used for the source log when invoking MUSE.",
    )
    parser.add_argument(
        "--tgt-label",
        default="tgt",
        help="Label used for the target log when invoking MUSE.",
    )
    parser.add_argument(
        "--embedding-method",
        choices=SUPPORTED_EMBEDDING_METHODS,
        default="Activity-Activitiy Co Occurrence Bag Of Words w_3",
        help=(
            "Embedding construction strategy. Methods mirror the options used in "
            "evaluation/evaluation_of_activity_distances/next_activity_prediction."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        help=(
            "Override the context window size for methods that depend on it. "
            "If omitted, the size is inferred from the method name (w_<size>) or "
            "defaults to 3."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used for randomized embedding methods (e.g., random uniform).",
    )
    parser.add_argument(
        "--muse-script",
        type=Path,
        help=(
            "Path to the MUSE unsupervised.py script. If provided, the script "
            "will be executed after preparing the embeddings."
        ),
    )
    parser.add_argument(
        "--muse-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Additional arguments passed verbatim to the MUSE unsupervised.py "
            "script."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the script.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    src_log = read_event_log(args.src_log)
    tgt_log = read_event_log(args.tgt_log)
    alphabet = compute_union_alphabet(src_log, tgt_log)

    resolved_window = args.window_size or extract_window_size(args.embedding_method) or WINDOW_SIZES[0]
    LOGGER.info(
        "Embedding method: %s (window_size=%d, random_seed=%d)",
        args.embedding_method,
        resolved_window,
        args.random_seed,
    )

    rng = np.random.default_rng(args.random_seed)
    src_embeddings, src_freq = compute_embeddings(
        src_log,
        alphabet,
        args.embedding_method,
        args.window_size,
        rng,
    )
    tgt_embeddings, tgt_freq = compute_embeddings(
        tgt_log,
        alphabet,
        args.embedding_method,
        args.window_size,
        rng,
    )

    src_file = output_dir / f"{args.src_label}_embeddings.vec"
    tgt_file = output_dir / f"{args.tgt_label}_embeddings.vec"

    src_activities = write_embeddings(src_embeddings, src_file)
    tgt_activities = write_embeddings(tgt_embeddings, tgt_file)

    prepare_metadata(
        output_dir,
        args.src_label,
        args.tgt_label,
        alphabet,
        {k: src_freq.get(k, 0) for k in src_activities},
        {k: tgt_freq.get(k, 0) for k in tgt_activities},
        src_file,
        tgt_file,
        args.embedding_method,
        args.window_size,
        args.random_seed,
    )

    if args.muse_script:
        run_muse_unsupervised(
            args.muse_script,
            src_file,
            tgt_file,
            args.src_label,
            args.tgt_label,
            args.muse_extra_args,
        )
    else:
        LOGGER.info(
            "MUSE script path not provided; skipping automatic alignment. "
            "You can run MUSE manually with the generated embedding files."
        )


if __name__ == "__main__":
    main()

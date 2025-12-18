"""
Single run: all uncertain count-based methods for window sizes [3,5,9].

This script is designed for the scenario where *trace realization enumeration* is the bottleneck.
We therefore enumerate deterministic realizations once and reuse the resulting expected counts
to build all count-based methods:

- Context interpretations: Seq, MSet
- Matrix types: AC, AA
- Post-processing: none, PMI, PPMI
- Window sizes: 3, 5, 9

Parallelism
-----------
We parallelize across uncertain traces using multiprocessing and use ~75% of available cores.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import multiprocessing as mp

import numpy as np

# Ensure repo root is on PYTHONPATH when running directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR
from distances.activity_distances.data_util.algorithm import get_cosine_distance_dict
from distances.activity_distances.pmi.pmi import (
    get_activity_activity_frequency_matrix_pmi,
    get_activity_context_frequency_matrix_pmi,
)
from distances.uncertain_activity_distances.data_util.uncertain_multiwindow_expected_counts import (
    compute_multiwindow_expected_counts,
)
from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import (
    PAD_TOKEN,
    compute_context_frequencies_from_expected_counts,
)
from uncertain_utils.uncertain_xes_reader import read_uncertain_xes


# =============================================================================
# Hardcoded configuration
# =============================================================================

XES_PATH = Path(ROOT_DIR) / "uncertain_event_logs" / "xes_uncertain_gt__no_na__p_ge_0_05.xes"

WINDOW_SIZES = [3, 5, 9]
NA_LABEL = "NA"

# Enumerate ALL realizations (exact) unless you set a cap
MAX_REALIZATIONS_PER_TRACE = None  # set e.g. 2000 for debugging
TOP_K = None
MIN_PROB = 0.0

# Multiprocessing
CPU = mp.cpu_count()
N_JOBS = max(1, int(CPU * 0.75))
MP_START_METHOD = "fork"  # macOS: "fork" is typically fastest; use "spawn" if needed

PRINT_PROGRESS = True

# Optional: exclude labels globally (e.g. {"NA"} if you keep NA in the log but want to ignore it)
EXCLUDE_ACTIVITIES = set([PAD_TOKEN])


def safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        pass


def build_ac_embeddings(expected_counts_for_w, alphabet):
    all_contexts = {ctx for cmap in expected_counts_for_w.values() for ctx in cmap.keys()}
    context_index = {ctx: i for i, ctx in enumerate(all_contexts)}
    embeddings = {a: np.zeros(len(all_contexts), dtype=float) for a in alphabet}
    for a, cmap in expected_counts_for_w.items():
        if a not in embeddings:
            continue
        for ctx, v in cmap.items():
            embeddings[a][context_index[ctx]] += float(v)
    return embeddings, context_index


def build_aa_embeddings_from_expected_counts(expected_counts_for_w, alphabet):
    # context -> {activity -> #(a,c)}
    context_to_counts = defaultdict(dict)
    for a, cmap in expected_counts_for_w.items():
        for ctx, v in cmap.items():
            context_to_counts[ctx][a] = float(v)

    activity_index = {a: i for i, a in enumerate(alphabet)}
    embeddings = {a: np.zeros(len(alphabet), dtype=float) for a in alphabet}

    for ctx, a_counts in context_to_counts.items():
        acts = list(a_counts.keys())
        for a in acts:
            row = embeddings.get(a)
            if row is None:
                continue
            for b in acts:
                row[activity_index[b]] += a_counts[a] + a_counts[b]

    return embeddings, activity_index


if __name__ == "__main__":
    print(f"Loading XES: {XES_PATH}")
    log = read_uncertain_xes(XES_PATH)
    print(f"log_name={log.log_name!r} traces={len(log.traces)} events={sum(len(t.events) for t in log.traces)}")
    print(f"window_sizes={WINDOW_SIZES}")
    print(f"n_jobs={N_JOBS}/{CPU} start_method={MP_START_METHOD}")
    print(f"top_k={TOP_K} min_prob={MIN_PROB} max_realizations_per_trace={MAX_REALIZATIONS_PER_TRACE}")

    progress = safe_print if PRINT_PROGRESS else None

    counts_seq_by_w, counts_mset_by_w, activity_freq = compute_multiwindow_expected_counts(
        log,
        window_sizes=WINDOW_SIZES,
        top_k=TOP_K,
        min_prob=MIN_PROB,
        na_label=NA_LABEL,
        max_realizations_per_trace=MAX_REALIZATIONS_PER_TRACE,
        exclude_activities=set(EXCLUDE_ACTIVITIES),
        progress=progress,
        n_jobs=N_JOBS,
        mp_start_method=MP_START_METHOD,
    )

    alphabet = sorted(activity_freq.keys())

    # Build all methods
    results = {}
    for w in WINDOW_SIZES:
        for ctx_name, counts_by_w in [("Seq", counts_seq_by_w), ("MSet", counts_mset_by_w)]:
            expected_counts = counts_by_w.get(w, {})

            # AC
            ac_emb, ctx_index = build_ac_embeddings(expected_counts, alphabet)
            ac_dist = get_cosine_distance_dict(ac_emb)
            ctx_freq = compute_context_frequencies_from_expected_counts(expected_counts)
            results[f"Uncertain AC {ctx_name} w_{w}"] = ac_dist
            results[f"Uncertain AC {ctx_name} PMI w_{w}"], _ = get_activity_context_frequency_matrix_pmi(
                ac_emb, activity_freq, ctx_freq, ctx_index, 0
            )
            results[f"Uncertain AC {ctx_name} PPMI w_{w}"], _ = get_activity_context_frequency_matrix_pmi(
                ac_emb, activity_freq, ctx_freq, ctx_index, 1
            )

            # AA
            aa_emb, act_index = build_aa_embeddings_from_expected_counts(expected_counts, alphabet)
            aa_dist = get_cosine_distance_dict(aa_emb)
            results[f"Uncertain AA {ctx_name} w_{w}"] = aa_dist
            results[f"Uncertain AA {ctx_name} PMI w_{w}"], _ = get_activity_activity_frequency_matrix_pmi(
                aa_emb, activity_freq, act_index, 0
            )
            results[f"Uncertain AA {ctx_name} PPMI w_{w}"], _ = get_activity_activity_frequency_matrix_pmi(
                aa_emb, activity_freq, act_index, 1
            )

    print(f"Done. Built {len(results)} distance matrices.")
    # Example: print one method's nearest neighbors for the most frequent activity
    if alphabet:
        q = max(activity_freq.items(), key=lambda kv: kv[1])[0]
        method = f"Uncertain AC Seq PPMI w_{WINDOW_SIZES[0]}"
        nested = defaultdict(dict)
        for (a, b), v in results[method].items():
            nested[a][b] = v
        nn = sorted(((b, d) for b, d in nested[q].items() if b != q), key=lambda kv: kv[1])[:10]
        print(f"Example method: {method}")
        print(f"Query activity: {q!r}")
        for b, d in nn:
            print(f"  {b!r:40s} dist={d:.4f}")




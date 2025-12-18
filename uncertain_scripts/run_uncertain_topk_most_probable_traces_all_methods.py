"""
Run: for each uncertain trace, generate up to K most probable realizations, report covered mass,
and build all count-based methods for window sizes 3/5/9.

This is the "top-K truncation" alternative to exact enumeration.
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
from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import (
    PAD_TOKEN,
    compute_context_frequencies_from_expected_counts,
)
from distances.uncertain_activity_distances.data_util.uncertain_multiwindow_topk_expected_counts import (
    compute_multiwindow_expected_counts_topk,
)
from uncertain_utils.uncertain_xes_reader import read_uncertain_xes


# =============================================================================
# Hardcoded configuration
# =============================================================================

# Use the original (non-thresholded) uncertain XES: we do NOT drop low-probability activities.
XES_PATH = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"
WINDOW_SIZES = [3, 5, 9]

# Top-K per uncertain trace
TOP_K_TRACES = 100_000

# Per-event option pruning: drop labels with p < threshold and renormalize remainder.
# Set to 0.0 to disable.
PROB_THRESHOLD = 0.10

# Multiprocessing: use 75% cores
CPU = mp.cpu_count()
N_JOBS = max(1, int(CPU * 0.75))
MP_START_METHOD = "fork"

# Progress + printing
PRINT_PROGRESS = True

# NA semantics
NA_LABEL = "NA"

# Exclusions
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
    print(f"window_sizes={WINDOW_SIZES} top_k_traces={TOP_K_TRACES}")
    print(f"n_jobs={N_JOBS}/{CPU} start_method={MP_START_METHOD}")

    progress = safe_print if PRINT_PROGRESS else None

    counts_seq_by_w, counts_mset_by_w, activity_freq, per_trace_mass = compute_multiwindow_expected_counts_topk(
        log,
        window_sizes=WINDOW_SIZES,
        top_k_traces=TOP_K_TRACES,
        prob_threshold=PROB_THRESHOLD,
        na_label=NA_LABEL,
        exclude_activities=set(EXCLUDE_ACTIVITIES),
        progress=progress,
        n_jobs=N_JOBS,
        mp_start_method=MP_START_METHOD,
    )

    # Print probability mass coverage summary
    per_trace_mass.sort(key=lambda x: x[0])
    masses = [m for _, _, m in per_trace_mass]
    avg_mass = sum(masses) / len(masses) if masses else 0.0
    print(f"\nTop-K probability mass coverage (K={TOP_K_TRACES}):")
    print(f" - traces: {len(masses)}")
    print(f" - avg_covered_mass: {avg_mass:.6f}")
    if masses:
        ms = sorted(masses)
        def q(p): 
            return ms[int(p*(len(ms)-1))]
        print(f" - covered_mass percentiles: p00={q(0.0):.6f} p25={q(0.25):.6f} p50={q(0.5):.6f} p75={q(0.75):.6f} p100={q(1.0):.6f}")

    print("\nFirst 10 traces (trace_idx, generated, covered_mass, remaining_mass):")
    for trace_idx, generated, mass in per_trace_mass[:10]:
        print(" ", (trace_idx, generated, float(mass), float(1.0 - mass)))

    worst = sorted(per_trace_mass, key=lambda x: x[2])[:10]
    print("\nWorst 10 traces by covered_mass (trace_idx, generated, covered_mass, remaining_mass):")
    for trace_idx, generated, mass in worst:
        print(" ", (trace_idx, generated, float(mass), float(1.0 - mass)))

    alphabet = sorted(activity_freq.keys())

    # Build all methods for all windows from the truncated expected counts
    results = {}
    for w in WINDOW_SIZES:
        for ctx_name, counts_by_w in [("Seq", counts_seq_by_w), ("MSet", counts_mset_by_w)]:
            expected_counts = counts_by_w.get(w, {})

            # AC
            ac_emb, ctx_index = build_ac_embeddings(expected_counts, alphabet)
            ctx_freq = compute_context_frequencies_from_expected_counts(expected_counts)
            results[f"Uncertain AC {ctx_name} w_{w}"] = get_cosine_distance_dict(ac_emb)
            results[f"Uncertain AC {ctx_name} PMI w_{w}"], _ = get_activity_context_frequency_matrix_pmi(
                ac_emb, activity_freq, ctx_freq, ctx_index, 0
            )
            results[f"Uncertain AC {ctx_name} PPMI w_{w}"], _ = get_activity_context_frequency_matrix_pmi(
                ac_emb, activity_freq, ctx_freq, ctx_index, 1
            )

            # AA
            aa_emb, act_index = build_aa_embeddings_from_expected_counts(expected_counts, alphabet)
            results[f"Uncertain AA {ctx_name} w_{w}"] = get_cosine_distance_dict(aa_emb)
            results[f"Uncertain AA {ctx_name} PMI w_{w}"], _ = get_activity_activity_frequency_matrix_pmi(
                aa_emb, activity_freq, act_index, 0
            )
            results[f"Uncertain AA {ctx_name} PPMI w_{w}"], _ = get_activity_activity_frequency_matrix_pmi(
                aa_emb, activity_freq, act_index, 1
            )

    print(f"\nDone. Built {len(results)} distance matrices from top-K truncated counts.")



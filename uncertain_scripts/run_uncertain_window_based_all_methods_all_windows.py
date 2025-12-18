"""
Run window-based uncertain count-based methods for window sizes [3,5,9] in one script.

This uses the NEW approach:
  enumerate all possible windows + their probabilities (with NA-skip semantics)
instead of enumerating all possible trace realizations.

Edit `XES_PATH` to point to your big dataset (e.g., the one with 371 traces) and run in PyCharm.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import gc

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR
from distances.activity_distances.data_util.algorithm import get_cosine_distance_dict
from distances.activity_distances.pmi.pmi import (
    get_activity_activity_frequency_matrix_pmi,
)
from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import (
    PAD_TOKEN,
)
from distances.uncertain_activity_distances.data_util.uncertain_sparse_math import (
    cosine_distance_matrix_sparse,
)
from distances.uncertain_activity_distances.data_util.uncertain_sparse_pmi import (
    ac_to_pmi_sparse,
)
from distances.uncertain_activity_distances.data_util.uncertain_window_based_multiwindow_expected_counts import (
    compute_expected_counts_window_based_multiwindow,
)
from uncertain_utils.uncertain_xes_reader import read_uncertain_xes


# =============================================================================
# Hardcoded config
# =============================================================================

# TODO: set this to your big uncertain XES (371 traces) path.
XES_PATH = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"

WINDOW_SIZES = [3, 5, 9]
NA_LABEL = "NA"
EXCLUDE = {"."}

# Per-event thresholding (drop labels with p < threshold, then renormalize remainder).
PROB_THRESHOLD = 0.05


def build_ac_embeddings(expected_counts, alphabet):
    # Keep embeddings sparse: activity -> {context -> weight}
    emb = {a: dict(expected_counts.get(a, {})) for a in alphabet}
    return emb


def build_aa_embeddings(expected_counts, alphabet):
    context_to_counts = defaultdict(dict)
    for a, cmap in expected_counts.items():
        for ctx, v in cmap.items():
            context_to_counts[ctx][a] = float(v)

    act_index = {a: i for i, a in enumerate(alphabet)}
    emb = {a: np.zeros(len(alphabet), dtype=float) for a in alphabet}
    for ctx, a_counts in context_to_counts.items():
        acts = list(a_counts.keys())
        for a in acts:
            row = emb.get(a)
            if row is None:
                continue
            for b in acts:
                row[act_index[b]] += a_counts[a] + a_counts[b]
    return emb, act_index


if __name__ == "__main__":
    print(f"Loading XES: {XES_PATH}")
    log = read_uncertain_xes(XES_PATH)
    print(f"log_name={log.log_name!r} traces={len(log.traces)} events={sum(len(t.events) for t in log.traces)}")

    # -------------------------------------------------------------------------
    # Memory optimization (1): compress activity labels to short IDs
    # Keep PAD and NA uncompressed to preserve semantics checks.
    # -------------------------------------------------------------------------
    all_labels = sorted(log.activities(exclude=set()))
    keep_as_is = {PAD_TOKEN, NA_LABEL}
    to_map = [a for a in all_labels if a not in keep_as_is]
    label_map = {a: str(i) for i, a in enumerate(to_map)}
    # Preserve special tokens
    label_map[PAD_TOKEN] = PAD_TOKEN
    label_map[NA_LABEL] = NA_LABEL
    inv_label_map = {v: k for k, v in label_map.items()}

    results = {}
    # We compute Seq and MSet counts separately (still far cheaper than trace-realization enumeration)
    for ctx_kind_name, ctx_kind in [("Seq", "seq"), ("MSet", "mset")]:
        # Important memory optimization:
        # computing all window sizes at once can keep multiple huge count maps resident in RAM.
        # Instead, compute one window size at a time and free intermediate structures.
        activity_freq = None

        for w in WINDOW_SIZES:
            counts_by_w, ctx_freq_by_w, activity_freq = compute_expected_counts_window_based_multiwindow(
                log,
                window_sizes=[w],
                context_kind=ctx_kind,
                na_label=NA_LABEL,
                prob_threshold=PROB_THRESHOLD,
                exclude_activities=set(EXCLUDE),
                progress=lambda m: print(m, flush=True),
                label_map=label_map,
            )

            expected_counts = counts_by_w[w]
            ctx_freq = ctx_freq_by_w[w]

            alphabet = sorted(activity_freq.keys())

            # AC
            ac_emb = build_ac_embeddings(expected_counts, alphabet)
            results[f"Uncertain-Window AC {ctx_kind_name} w_{w}"] = cosine_distance_matrix_sparse(ac_emb)

            ac_pmi = ac_to_pmi_sparse(ac_emb, activity_freq=activity_freq, context_freq=ctx_freq, ppmi=False)
            results[f"Uncertain-Window AC {ctx_kind_name} PMI w_{w}"] = cosine_distance_matrix_sparse(ac_pmi)

            ac_ppmi = ac_to_pmi_sparse(ac_emb, activity_freq=activity_freq, context_freq=ctx_freq, ppmi=True)
            results[f"Uncertain-Window AC {ctx_kind_name} PPMI w_{w}"] = cosine_distance_matrix_sparse(ac_ppmi)

            # AA
            aa_emb, act_index = build_aa_embeddings(expected_counts, alphabet)
            results[f"Uncertain-Window AA {ctx_kind_name} w_{w}"] = get_cosine_distance_dict(aa_emb)
            results[f"Uncertain-Window AA {ctx_kind_name} PMI w_{w}"], _ = get_activity_activity_frequency_matrix_pmi(
                aa_emb, activity_freq, act_index, 0
            )
            results[f"Uncertain-Window AA {ctx_kind_name} PPMI w_{w}"], _ = get_activity_activity_frequency_matrix_pmi(
                aa_emb, activity_freq, act_index, 1
            )

            # Free big intermediates ASAP to reduce peak RSS
            del counts_by_w
            del expected_counts
            del ac_emb, ctx_freq, ac_pmi, ac_ppmi
            del aa_emb, act_index
            del ctx_freq_by_w
            gc.collect()

    # Map result keys back to original labels (small |A| so this is cheap)
    mapped_results = {}
    for name, dist in results.items():
        mapped = {}
        for (a, b), v in dist.items():
            mapped[(inv_label_map.get(a, a), inv_label_map.get(b, b))] = v
        mapped_results[name] = mapped

    print(f"Done. Built {len(mapped_results)} distance matrices (window-based).")



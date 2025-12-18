"""
Memory-safe window-based runner: run each (context_kind, window_size) pass in a subprocess.

Why this exists
---------------
Even if we `del` large dicts and call `gc.collect()`, Python often does not return
freed memory to the OS, so RSS can stay at the *peak* (e.g., 25+ GB) after a heavy pass.

Running each pass in its own subprocess makes the OS reclaim memory when the subprocess exits.
This typically keeps peak RSS bounded in the parent process.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import multiprocessing as mp

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR


# =============================================================================
# Hardcoded config
# =============================================================================

XES_PATH = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"
WINDOW_SIZES = [3, 5, 9]
NA_LABEL = "NA"
EXCLUDE = {"."}
PROB_THRESHOLD = 0.05

MP_START_METHOD = "spawn"  # macOS-friendly


def _run_one_pass(args):
    """
    Worker: compute window-based counts and build all 6 matrices for one (ctx_kind, w) pair.
    Returns dict[name -> distance_dict] with ORIGINAL activity names as keys.
    """
    (xes_path_str, ctx_kind_name, ctx_kind, w, prob_threshold) = args

    # Local imports inside worker (safe under spawn)
    from uncertain_utils.uncertain_xes_reader import read_uncertain_xes
    from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import PAD_TOKEN
    from distances.uncertain_activity_distances.data_util.uncertain_sparse_math import cosine_distance_matrix_sparse
    from distances.uncertain_activity_distances.data_util.uncertain_sparse_pmi import ac_to_pmi_sparse
    from distances.activity_distances.data_util.algorithm import get_cosine_distance_dict
    from distances.activity_distances.pmi.pmi import get_activity_activity_frequency_matrix_pmi
    from distances.uncertain_activity_distances.data_util.uncertain_window_based_multiwindow_expected_counts import (
        compute_expected_counts_window_based_multiwindow,
    )

    log = read_uncertain_xes(Path(xes_path_str))

    # Label compression inside worker to reduce key sizes during counting
    all_labels = sorted(log.activities(exclude=set()))
    keep_as_is = {PAD_TOKEN, NA_LABEL}
    to_map = [a for a in all_labels if a not in keep_as_is]
    label_map = {a: str(i) for i, a in enumerate(to_map)}
    label_map[PAD_TOKEN] = PAD_TOKEN
    label_map[NA_LABEL] = NA_LABEL
    inv_label_map = {v: k for k, v in label_map.items()}

    counts_by_w, ctx_freq_by_w, activity_freq = compute_expected_counts_window_based_multiwindow(
        log,
        window_sizes=[w],
        context_kind=ctx_kind,
        na_label=NA_LABEL,
        prob_threshold=prob_threshold,
        exclude_activities=set(EXCLUDE),
        progress=None,
        label_map=label_map,
    )
    expected_counts = counts_by_w[w]
    ctx_freq = ctx_freq_by_w[w]

    alphabet = sorted(activity_freq.keys())

    out = {}

    # AC (sparse)
    ac_emb = {a: dict(expected_counts.get(a, {})) for a in alphabet}
    out[f"Uncertain-Window AC {ctx_kind_name} w_{w}"] = cosine_distance_matrix_sparse(ac_emb)
    ac_pmi = ac_to_pmi_sparse(ac_emb, activity_freq=activity_freq, context_freq=ctx_freq, ppmi=False)
    out[f"Uncertain-Window AC {ctx_kind_name} PMI w_{w}"] = cosine_distance_matrix_sparse(ac_pmi)
    ac_ppmi = ac_to_pmi_sparse(ac_emb, activity_freq=activity_freq, context_freq=ctx_freq, ppmi=True)
    out[f"Uncertain-Window AC {ctx_kind_name} PPMI w_{w}"] = cosine_distance_matrix_sparse(ac_ppmi)

    # AA (dense small |A|)
    context_to_counts = defaultdict(dict)
    for a, cmap in expected_counts.items():
        for ctx, v in cmap.items():
            context_to_counts[ctx][a] = float(v)
    act_index = {a: i for i, a in enumerate(alphabet)}
    aa_emb = {a: np.zeros(len(alphabet), dtype=float) for a in alphabet}
    for ctx, a_counts in context_to_counts.items():
        acts = list(a_counts.keys())
        for a in acts:
            row = aa_emb.get(a)
            if row is None:
                continue
            for b in acts:
                row[act_index[b]] += a_counts[a] + a_counts[b]
    out[f"Uncertain-Window AA {ctx_kind_name} w_{w}"] = get_cosine_distance_dict(aa_emb)
    out[f"Uncertain-Window AA {ctx_kind_name} PMI w_{w}"], _ = get_activity_activity_frequency_matrix_pmi(
        aa_emb, activity_freq, act_index, 0
    )
    out[f"Uncertain-Window AA {ctx_kind_name} PPMI w_{w}"], _ = get_activity_activity_frequency_matrix_pmi(
        aa_emb, activity_freq, act_index, 1
    )

    # map back to original labels
    mapped_out = {}
    for name, dist in out.items():
        mapped = {}
        for (a, b), v in dist.items():
            mapped[(inv_label_map.get(a, a), inv_label_map.get(b, b))] = v
        mapped_out[name] = mapped
    return mapped_out


if __name__ == "__main__":
    print(f"Memory-safe subprocess runner")
    print(f"xes={XES_PATH}")
    print(f"window_sizes={WINDOW_SIZES} prob_threshold={PROB_THRESHOLD}")

    ctx = mp.get_context(MP_START_METHOD)
    tasks = []
    for ctx_kind_name, ctx_kind in [("Seq", "seq"), ("MSet", "mset")]:
        for w in WINDOW_SIZES:
            tasks.append((str(XES_PATH), ctx_kind_name, ctx_kind, w, PROB_THRESHOLD))

    results = {}
    for (xes_path_str, ck_name, ck, w, thr) in tasks:
        print(f"[subprocess] start pass: kind={ck_name} w={w} thr={thr}", flush=True)
        with ctx.Pool(processes=1) as pool:
            out = pool.map(_run_one_pass, [(xes_path_str, ck_name, ck, w, thr)])[0]
        results.update(out)
        print(f"[subprocess] done pass: kind={ck_name} w={w} (matrices_added={len(out)})", flush=True)

    print(f"Done. Built {len(results)} distance matrices (window-based, subprocess mode).")




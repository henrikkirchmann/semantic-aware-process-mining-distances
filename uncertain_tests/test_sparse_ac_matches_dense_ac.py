import math
import unittest

import numpy as np

from uncertain_utils.uncertain_xes_reader import UncertainEvent, UncertainEventLog, UncertainTrace
from distances.activity_distances.pmi.pmi import get_activity_context_frequency_matrix_pmi
from distances.activity_distances.data_util.algorithm import get_cosine_distance_dict
from distances.uncertain_activity_distances.data_util.uncertain_sparse_math import cosine_distance_matrix_sparse
from distances.uncertain_activity_distances.data_util.uncertain_sparse_pmi import ac_to_pmi_sparse
from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import compute_context_frequencies_from_expected_counts
from distances.uncertain_activity_distances.data_util.uncertain_window_based_multiwindow_expected_counts import (
    compute_expected_counts_window_based_multiwindow,
)


def _mk_log_one_trace(event_dists, *, trace_id="0", case_name="case"):
    events = [UncertainEvent(activity_probs=d, attributes={}) for d in event_dists]
    tr = UncertainTrace(trace_id=trace_id, case_name=case_name, events=events, attributes={})
    return UncertainEventLog(traces=[tr], log_name="synthetic")


def _dense_ac_embeddings(expected_counts, alphabet):
    all_contexts = {ctx for cmap in expected_counts.values() for ctx in cmap.keys()}
    context_index = {ctx: i for i, ctx in enumerate(all_contexts)}
    emb = {a: np.zeros(len(all_contexts), dtype=float) for a in alphabet}
    for a, cmap in expected_counts.items():
        if a not in emb:
            continue
        for ctx, v in cmap.items():
            emb[a][context_index[ctx]] += float(v)
    return emb, context_index


def _assert_pair_dict_close(testcase, d1, d2, *, rel=1e-9, abs_=1e-9):
    testcase.assertEqual(set(d1.keys()), set(d2.keys()))
    for k in d1.keys():
        testcase.assertTrue(
            math.isclose(float(d1[k]), float(d2[k]), rel_tol=rel, abs_tol=abs_),
            f"Mismatch at {k}: {d1[k]} vs {d2[k]}",
        )


class TestSparseACMatchesDenseAC(unittest.TestCase):
    def test_sparse_cosine_matches_dense(self):
        log = _mk_log_one_trace(
            [
                {"a": 0.6, "b": 0.4},
                {"c": 1.0},
                {"d": 0.5, "e": 0.5},
                {"f": 1.0},
            ]
        )

        counts_by_w, act_freq = compute_expected_counts_window_based_multiwindow(
            log,
            window_sizes=[3],
            context_kind="seq",
            na_label="NA",
            prob_threshold=0.0,
            exclude_activities={"."},
        )
        expected_counts = counts_by_w[3]
        alphabet = sorted(act_freq.keys())

        # Dense cosine
        dense_emb, ctx_index = _dense_ac_embeddings(expected_counts, alphabet)
        dense_dist = get_cosine_distance_dict(dense_emb)

        # Sparse cosine
        sparse_emb = {a: dict(expected_counts.get(a, {})) for a in alphabet}
        sparse_dist = cosine_distance_matrix_sparse(sparse_emb)

        _assert_pair_dict_close(self, dense_dist, sparse_dist, rel=1e-9, abs_=1e-9)

    def test_sparse_pmi_matches_dense_pmi(self):
        log = _mk_log_one_trace(
            [
                {"a": 0.6, "b": 0.4},
                {"c": 1.0},
                {"d": 0.5, "e": 0.5},
                {"f": 1.0},
            ]
        )

        counts_by_w, act_freq = compute_expected_counts_window_based_multiwindow(
            log,
            window_sizes=[3],
            context_kind="seq",
            na_label="NA",
            prob_threshold=0.0,
            exclude_activities={"."},
        )
        expected_counts = counts_by_w[3]
        alphabet = sorted(act_freq.keys())

        # Dense
        dense_emb, ctx_index = _dense_ac_embeddings(expected_counts, alphabet)
        ctx_freq = compute_context_frequencies_from_expected_counts(expected_counts)
        dense_pmi_dist, dense_pmi_emb = get_activity_context_frequency_matrix_pmi(dense_emb, act_freq, ctx_freq, ctx_index, 0)
        dense_ppmi_dist, dense_ppmi_emb = get_activity_context_frequency_matrix_pmi(dense_emb, act_freq, ctx_freq, ctx_index, 1)

        # Sparse
        sparse_emb = {a: dict(expected_counts.get(a, {})) for a in alphabet}
        sparse_pmi_emb = ac_to_pmi_sparse(sparse_emb, activity_freq=act_freq, context_freq=ctx_freq, ppmi=False)
        sparse_ppmi_emb = ac_to_pmi_sparse(sparse_emb, activity_freq=act_freq, context_freq=ctx_freq, ppmi=True)
        sparse_pmi_dist = cosine_distance_matrix_sparse(sparse_pmi_emb)
        sparse_ppmi_dist = cosine_distance_matrix_sparse(sparse_ppmi_emb)

        _assert_pair_dict_close(self, dense_pmi_dist, sparse_pmi_dist, rel=1e-9, abs_=1e-9)
        _assert_pair_dict_close(self, dense_ppmi_dist, sparse_ppmi_dist, rel=1e-9, abs_=1e-9)


if __name__ == "__main__":
    unittest.main()




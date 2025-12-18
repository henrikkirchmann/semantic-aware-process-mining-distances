import math
import unittest

from uncertain_utils.uncertain_xes_reader import UncertainEvent, UncertainEventLog, UncertainTrace
from distances.uncertain_activity_distances.data_util.uncertain_expected_counts import (
    compute_expected_context_counts_and_activity_frequencies,
)
from distances.uncertain_activity_distances.data_util.uncertain_window_based_expected_counts import (
    compute_expected_counts_window_based,
)


def _assert_nested_float_dicts_close(testcase, a, b, *, rel=1e-9, abs_=1e-9):
    """
    Compare dict[str, dict[context, float]] with tolerance.
    """
    testcase.assertEqual(set(a.keys()), set(b.keys()), "Outer keys (activities) differ")
    for act in a.keys():
        testcase.assertEqual(set(a[act].keys()), set(b[act].keys()), f"Context keys differ for activity={act!r}")
        for ctx in a[act].keys():
            va = float(a[act][ctx])
            vb = float(b[act][ctx])
            testcase.assertTrue(
                math.isclose(va, vb, rel_tol=rel, abs_tol=abs_),
                f"Value mismatch for activity={act!r} ctx={ctx!r}: {va} vs {vb}",
            )


def _assert_float_dicts_close(testcase, a, b, *, rel=1e-9, abs_=1e-9):
    testcase.assertEqual(set(a.keys()), set(b.keys()), "Keys differ")
    for k in a.keys():
        testcase.assertTrue(
            math.isclose(float(a[k]), float(b[k]), rel_tol=rel, abs_tol=abs_),
            f"Value mismatch for key={k!r}: {a[k]} vs {b[k]}",
        )


def _mk_log_one_trace(event_dists, *, trace_id="0", case_name="case"):
    events = [UncertainEvent(activity_probs=d, attributes={}) for d in event_dists]
    tr = UncertainTrace(trace_id=trace_id, case_name=case_name, events=events, attributes={})
    return UncertainEventLog(traces=[tr], log_name="synthetic")


class TestWindowBasedVsTraceBased(unittest.TestCase):
    """
    These tests check that:
    - Old method (trace-based): enumerate deterministic traces, then slide windows
    - New method (window-based): enumerate deterministic windows directly
    yield the same expected counts under the same NA-skip semantics.
    """

    def test_no_na_window3_seq(self):
        # Simple 3-event trace with small supports, no NA.
        log = _mk_log_one_trace(
            [
                {"a": 0.6, "b": 0.4},
                {"c": 0.5, "d": 0.5},
                {"e": 1.0},
            ]
        )

        trace_counts, trace_freq = compute_expected_context_counts_and_activity_frequencies(
            log,
            ngram_size=3,
            context_kind="seq",
            na_label="NA",
            exclude_activities={"."},
            n_jobs=1,
        )
        win_counts, win_freq = compute_expected_counts_window_based(
            log,
            window_size=3,
            context_kind="seq",
            na_label="NA",
            exclude_activities={"."},
        )

        _assert_nested_float_dicts_close(self, trace_counts, win_counts)
        _assert_float_dicts_close(self, trace_freq, win_freq)

    def test_with_na_window5_seq(self):
        # NA can appear and means "event absent". Window size 5 forces skipping/expanding.
        log = _mk_log_one_trace(
            [
                {"x": 0.5, "NA": 0.5},
                {"a": 0.7, "NA": 0.3},
                {"b": 1.0},
                {"c": 0.6, "NA": 0.4},
                {"y": 0.2, "z": 0.8},
            ]
        )

        trace_counts, trace_freq = compute_expected_context_counts_and_activity_frequencies(
            log,
            ngram_size=5,
            context_kind="seq",
            na_label="NA",
            exclude_activities={"."},
            n_jobs=1,
        )
        win_counts, win_freq = compute_expected_counts_window_based(
            log,
            window_size=5,
            context_kind="seq",
            na_label="NA",
            exclude_activities={"."},
        )

        _assert_nested_float_dicts_close(self, trace_counts, win_counts)
        _assert_float_dicts_close(self, trace_freq, win_freq)

    def test_with_na_window5_mset(self):
        log = _mk_log_one_trace(
            [
                {"x": 0.5, "NA": 0.5},
                {"a": 0.7, "NA": 0.3},
                {"b": 1.0},
                {"c": 0.6, "NA": 0.4},
                {"y": 0.2, "z": 0.8},
            ]
        )

        trace_counts, trace_freq = compute_expected_context_counts_and_activity_frequencies(
            log,
            ngram_size=5,
            context_kind="mset",
            na_label="NA",
            exclude_activities={"."},
            n_jobs=1,
        )
        win_counts, win_freq = compute_expected_counts_window_based(
            log,
            window_size=5,
            context_kind="mset",
            na_label="NA",
            exclude_activities={"."},
        )

        _assert_nested_float_dicts_close(self, trace_counts, win_counts)
        _assert_float_dicts_close(self, trace_freq, win_freq)


if __name__ == "__main__":
    unittest.main()




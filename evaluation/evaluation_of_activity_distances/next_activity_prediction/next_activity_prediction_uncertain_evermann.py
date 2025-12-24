"""
Uncertain next-activity prediction benchmark (IKEA ASM).

Implements the "option (1)" benchmark discussed in the paper draft:
  - Input traces: predicted merged segments from the action recognition model (uncertain events)
  - Target next activity: first non-NA ground-truth label AFTER the segment ends
  - Model: Evermann-style next-activity predictor (single LSTM over activity representations)

Inputs are read from:
  <ROOT_DIR>/uncertain_event_data/ikea_asm/split=test/model=<model_id>/
    - segments_pred.csv   (one row per predicted segment, includes avg_probs_json)
    - frames.csv          (one row per frame, includes gt_label_name)

PyCharm-friendly: this module exposes `run_uncertain_next_activity_prediction(...)`
and a separate runner script can call it with hardcoded settings.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from definitions import ROOT_DIR
from uncertain_utils.uncertain_xes_reader import UncertainEvent, UncertainEventLog, UncertainTrace

from distances.uncertain_activity_distances.uncertain_act2vec.uncertain_act2vec import (
    UncertainAct2VecConfig,
    get_uncertain_act2vec_distance_matrix,
)
from distances.uncertain_activity_distances.data_util.uncertain_window_based_multiwindow_expected_counts import (
    compute_expected_counts_window_based_multiwindow,
)
from distances.uncertain_activity_distances.data_util.uncertain_sparse_pmi import ac_to_pmi_sparse
from distances.activity_distances.pmi.pmi import get_activity_activity_frequency_matrix_pmi


def _pick_torch_device() -> str:
    """
    Pick best available PyTorch device (cuda -> mps -> cpu).
    """
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and getattr(mps, "is_available", None) and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    start_t: int
    end_t: int
    duration: int
    probs: Dict[str, float]  # includes "NA" if present in source


@dataclass(frozen=True)
class CaseData:
    segments: List[Segment]
    # Ground-truth can be represented either per-frame (gt_ts + gt_label + lookup),
    # or as segments (gt_segments) if frames.csv is not available.
    gt_ts: Optional[np.ndarray]  # int timestamps (ascending), or None
    gt_label: Optional[List[str]]  # same length as gt_ts, or None
    next_non_na_from_pos: Optional[List[Optional[str]]]  # length = len(gt_label)+1, or None
    gt_segments: Optional[List[Tuple[int, int, str]]]  # list of (start_t, end_t, label_name), sorted by start_t


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _topk_renorm(dist: Dict[str, float], *, k: int) -> Dict[str, float]:
    """Keep top-k by prob and renormalize to sum to 1.0 (if possible)."""
    items = [(str(a), _safe_float(p)) for a, p in dist.items() if p is not None and _safe_float(p) > 0.0]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1], reverse=True)
    items = items[: max(1, int(k))]
    out = {a: float(p) for a, p in items}
    s = float(sum(out.values()))
    if s > 0:
        out = {a: float(p) / s for a, p in out.items()}
    return out


def _drop_na_and_renorm(dist: Dict[str, float], *, na_label: str = "NA") -> Dict[str, float]:
    out = {a: float(p) for a, p in dist.items() if a != na_label and p is not None and float(p) > 0.0}
    s = float(sum(out.values()))
    if s > 0:
        out = {a: float(p) / s for a, p in out.items()}
    else:
        out = {}
    return out


def _argmax_label(dist: Dict[str, float]) -> Optional[str]:
    items = [(str(a), _safe_float(p)) for a, p in dist.items() if p is not None and _safe_float(p) > 0.0]
    if not items:
        return None
    return max(items, key=lambda kv: kv[1])[0]


def _determinize_segment_probs_top1(
    probs: Dict[str, float],
    *,
    na_label: str = "NA",
) -> Dict[str, float]:
    """
    Determinize a segment distribution to a single label with prob 1.0.

    We pick the argmax over non-NA labels if possible; if all mass is NA (or empty),
    we fall back to NA (non-existence) with probability 1.0.
    """
    d = _drop_na_and_renorm(probs, na_label=na_label)
    a = _argmax_label(d)
    if a is None:
        return {na_label: 1.0}
    return {str(a): 1.0}


def _build_next_non_na_lookup(gt_ts: np.ndarray, gt_label: List[str], *, na_label: str) -> List[Optional[str]]:
    """
    For each frame index j, return the first non-NA label at position >= j.
    If none exists, value is None.
    """
    n = len(gt_label)
    nxt: List[Optional[str]] = [None] * (n + 1)
    cur: Optional[str] = None
    for i in range(n - 1, -1, -1):
        lab = gt_label[i]
        if lab != na_label:
            cur = lab
        nxt[i] = cur
    nxt[n] = None
    return nxt


def _first_non_na_after_end(case: CaseData, *, end_t: int, na_label: str) -> Optional[str]:
    """
    Return the first non-NA GT label with timestamp strictly greater than `end_t`.
    """
    # Segment-level GT fallback (no per-frame file available)
    if case.gt_segments is not None:
        t = int(end_t) + 1  # strictly greater than end_t
        segs = case.gt_segments
        if not segs:
            return None

        starts = [s for (s, _e, _lab) in segs]
        i = int(np.searchsorted(np.asarray(starts, dtype=np.int64), t, side="right")) - 1
        if i < 0:
            i = 0

        # If t lies within current segment, return it if non-NA; otherwise scan forward.
        s0, e0, lab0 = segs[i]
        if s0 <= t <= e0:
            if lab0 != na_label:
                return lab0
            j = i + 1
        else:
            # t is after seg i ends (or before it starts); next segment is i+1 or the first with start > t
            j = i + 1
            while j < len(segs) and segs[j][0] <= t and segs[j][1] < t:
                j += 1

        while j < len(segs):
            _s, _e, lab = segs[j]
            if lab != na_label:
                return lab
            j += 1
        return None

    # Per-frame GT (original path)
    ts = case.gt_ts if case.gt_ts is not None else np.asarray([], dtype=np.int64)
    nxt = case.next_non_na_from_pos or []
    # first index with ts > end_t
    j = int(np.searchsorted(ts, int(end_t), side="right"))
    return nxt[j] if 0 <= j < len(nxt) else None


# -----------------------------------------------------------------------------
# Reading IKEA ASM split=test data
# -----------------------------------------------------------------------------


def load_ikea_split_test_model(
    *,
    model_id: str,
    # If provided, interpret as the repository root (or any directory that contains `uncertain_event_data/`).
    data_root: Optional[Path] = None,
    # Backwards-compat override: directly provide the model directory that contains `segments_pred.csv` and `frames.csv`.
    base_dir: Optional[Path] = None,
    na_label: str = "NA",
    top_k_event: int = 3,
) -> Dict[int, CaseData]:
    """
    Load segments + GT frames for one model in IKEA ASM split=test.
    Returns dict[case_id] -> CaseData.

    We cap each segment distribution to top_k_event labels and renormalize (benchmark setting).
    """
    if base_dir is not None:
        root = Path(base_dir)
    else:
        root_base = Path(data_root) if data_root is not None else Path(ROOT_DIR)
        root = root_base / "uncertain_event_data" / "ikea_asm" / "split=test" / f"model={model_id}"
    seg_csv = root / "segments_pred.csv"
    frames_csv = root / "frames.csv"
    if not seg_csv.exists():
        raise FileNotFoundError(str(seg_csv))

    # segments: we need avg_probs_json; file is small enough.
    df_seg = pd.read_csv(seg_csv)
    required_cols = {"case_id", "start_timestamp", "end_timestamp", "duration_frames", "avg_probs_json"}
    missing = required_cols - set(df_seg.columns)
    if missing:
        raise ValueError(f"segments_pred.csv missing columns: {sorted(missing)}")

    # Ground truth: prefer per-frame (frames.csv). If missing (common on Windows due to large file),
    # fall back to segment-level GT from segments_gt.csv.
    gt_by_case: Dict[int, Tuple[Optional[np.ndarray], Optional[List[str]], Optional[List[Optional[str]]], Optional[List[Tuple[int, int, str]]]]] = {}
    if frames_csv.exists():
        # frames: avoid loading probs_json (huge). Only need gt label per timestamp.
        df_frames = pd.read_csv(
            frames_csv,
            usecols=["case_id", "timestamp", "gt_label_name"],
            dtype={"case_id": "int32", "timestamp": "int32", "gt_label_name": "string"},
        )

        for cid, grp in df_frames.groupby("case_id", sort=False):
            g = grp.sort_values("timestamp", kind="mergesort")
            ts = g["timestamp"].to_numpy(dtype=np.int32)
            labs = [str(x) for x in g["gt_label_name"].tolist()]
            nxt = _build_next_non_na_lookup(ts, labs, na_label=na_label)
            gt_by_case[int(cid)] = (ts, labs, nxt, None)
    else:
        seg_gt_csv = root / "segments_gt.csv"
        if not seg_gt_csv.exists():
            raise FileNotFoundError(str(frames_csv))
        df_gt = pd.read_csv(seg_gt_csv)
        req = {"case_id", "start_timestamp", "end_timestamp", "gt_label_name"}
        missing_gt = req - set(df_gt.columns)
        if missing_gt:
            raise ValueError(f"segments_gt.csv missing columns: {sorted(missing_gt)}")
        for cid, grp in df_gt.groupby("case_id", sort=False):
            g = grp.sort_values("start_timestamp", kind="mergesort")
            segs = [(int(r.start_timestamp), int(r.end_timestamp), str(r.gt_label_name)) for r in g.itertuples(index=False)]
            gt_by_case[int(cid)] = (None, None, None, segs)

    # Build segments by case
    seg_by_case: Dict[int, List[Segment]] = {}
    for row in df_seg.itertuples(index=False):
        cid = int(getattr(row, "case_id"))
        probs = json.loads(getattr(row, "avg_probs_json"))
        probs = {str(a): float(p) for a, p in probs.items()}
        probs = _topk_renorm(probs, k=int(top_k_event))
        seg_by_case.setdefault(cid, []).append(
            Segment(
                start_t=int(getattr(row, "start_timestamp")),
                end_t=int(getattr(row, "end_timestamp")),
                duration=int(getattr(row, "duration_frames")),
                probs=probs,
            )
        )

    out: Dict[int, CaseData] = {}
    for cid, segs in seg_by_case.items():
        if cid not in gt_by_case:
            continue
        segs_sorted = sorted(segs, key=lambda s: (s.start_t, s.end_t))
        ts, labs, nxt, segs_gt = gt_by_case[cid]
        out[int(cid)] = CaseData(
            segments=segs_sorted,
            gt_ts=ts,
            gt_label=labs,
            next_non_na_from_pos=nxt,
            gt_segments=segs_gt,
        )

    return out


def build_uncertain_event_log_from_cases(
    cases: Dict[int, CaseData],
    *,
    na_label: str = "NA",
    determinize_top1: bool = False,
) -> UncertainEventLog:
    """
    Convert cases to UncertainEventLog for embedding training (one event per predicted segment).
    """
    traces: List[UncertainTrace] = []
    for cid, case in cases.items():
        events = []
        for seg in case.segments:
            probs = dict(seg.probs)
            if determinize_top1:
                probs = _determinize_segment_probs_top1(probs, na_label=na_label)
            events.append(UncertainEvent(activity_probs=probs, attributes={"segment:end_timestamp": seg.end_t}))
        traces.append(UncertainTrace(trace_id=str(cid), case_name=str(cid), events=events, attributes={}))
    return UncertainEventLog(traces=traces, log_name="ikea_asm_pred_merged")


# -----------------------------------------------------------------------------
# Embedding computation (uncertain)
# -----------------------------------------------------------------------------


def _compute_uncertain_embeddings(
    log: UncertainEventLog,
    *,
    embedding_method: str,
    window_size: int,
    na_label: str = "NA",
) -> Dict[str, np.ndarray]:
    """
    Return activity->vector for one embedding method.

    Supported:
    - Uncertain AA/AC Seq/MSet (None/PMI/PPMI): window-based expected counts
    - Uncertain act2vec CBOW / Skip-gram: uncertain act2vec embeddings
    """
    method = embedding_method.strip()

    # Neural
    if method in ("Uncertain act2vec CBOW", "Uncertain act2vec Skip-gram"):
        sg = 0 if method.endswith("CBOW") else 1
        device = _pick_torch_device()
        cfg = UncertainAct2VecConfig(
            window_size=int(window_size),
            embedding_dim=16,
            epochs=10,
            batch_size=256,
            start_alpha=0.025,
            alpha_decay_per_epoch=0.002,
            min_alpha=0.0001,
            seed=0,
            device=device,
            training_mode="window_realizations",
            prob_threshold=0.0,
            top_k=None,
            min_prob=0.0,
            na_label=na_label,
            pad_token=".",
            drop_na_and_renormalize=False,
            exclude_activities=None,
            negative=5,
            negative_sampling_exponent=0.75,
            progress_every_samples=0,
        )
        _dist, emb, _dbg = get_uncertain_act2vec_distance_matrix(log, sg=sg, config=cfg, progress=None)
        return {str(a): np.asarray(v, dtype=np.float32) for a, v in emb.items()}

    # Count-based (window-based)
    if not method.startswith("Uncertain "):
        raise ValueError(f"Unknown embedding_method: {method!r}")
    matrix_type = "AA" if " AA " in f" {method} " else "AC"
    context_kind = "seq" if " Seq" in method else "mset"
    post = "none"
    if method.endswith(" PPMI"):
        post = "ppmi"
    elif method.endswith(" PMI"):
        post = "pmi"

    counts_by_w, ctx_freq_by_w, act_freq = compute_expected_counts_window_based_multiwindow(
        log,
        window_sizes=[int(window_size)],
        context_kind=context_kind,
        na_label=na_label,
        prob_threshold=0.0,
        progress=None,
    )
    counts = counts_by_w[int(window_size)]  # activity -> context -> expected count

    if matrix_type == "AA":
        alphabet = sorted(act_freq.keys())
        idx = {a: i for i, a in enumerate(alphabet)}
        aa_emb: Dict[str, np.ndarray] = {a: np.zeros(len(alphabet), dtype=float) for a in alphabet}
        # invert contexts
        context_to_counts: Dict[object, Dict[str, float]] = {}
        for a, cmap in counts.items():
            for ctx, v in cmap.items():
                context_to_counts.setdefault(ctx, {})[a] = float(v)
        for ctx, a_counts in context_to_counts.items():
            acts = list(a_counts.keys())
            for a in acts:
                row = aa_emb[a]
                for b in acts:
                    row[idx[b]] += a_counts[a] + a_counts[b]

        if post == "pmi":
            _d, emb = get_activity_activity_frequency_matrix_pmi(aa_emb, act_freq, idx, 0)
            return {a: np.asarray(v, dtype=np.float32) for a, v in emb.items()}
        if post == "ppmi":
            _d, emb = get_activity_activity_frequency_matrix_pmi(aa_emb, act_freq, idx, 1)
            return {a: np.asarray(v, dtype=np.float32) for a, v in emb.items()}
        return {a: np.asarray(v, dtype=np.float32) for a, v in aa_emb.items()}

    # AC: build sparse dict embeddings, optionally PMI/PPMI, then densify over observed contexts.
    ac_sparse: Dict[str, Dict[object, float]] = {a: dict(cmap) for a, cmap in counts.items()}
    if post == "pmi":
        ac_sparse = ac_to_pmi_sparse(ac_sparse, activity_freq=act_freq, context_freq=ctx_freq_by_w[int(window_size)], ppmi=False)
    elif post == "ppmi":
        ac_sparse = ac_to_pmi_sparse(ac_sparse, activity_freq=act_freq, context_freq=ctx_freq_by_w[int(window_size)], ppmi=True)

    # context index over all contexts
    ctx_set = set()
    for cmap in ac_sparse.values():
        ctx_set.update(cmap.keys())
    ctx_list = sorted(ctx_set, key=lambda x: str(x))
    ctx_idx = {c: i for i, c in enumerate(ctx_list)}
    dim = len(ctx_list)

    emb: Dict[str, np.ndarray] = {}
    for a, cmap in ac_sparse.items():
        v = np.zeros(dim, dtype=np.float32)
        for c, val in cmap.items():
            j = ctx_idx.get(c)
            if j is not None:
                v[j] = float(val)
        emb[str(a)] = v
    return emb


# -----------------------------------------------------------------------------
# Input representations for Evermann
# -----------------------------------------------------------------------------


def _segment_vector_expected_embedding(
    seg: Segment,
    *,
    emb: Dict[str, np.ndarray],
    alphabet: List[str],
    na_label: str,
) -> np.ndarray:
    """
    Expected embedding: drop NA + renorm, then Σ p(a) v(a).
    """
    d = _drop_na_and_renorm(seg.probs, na_label=na_label)
    if not d:
        # all mass on NA or empty
        dim = len(next(iter(emb.values()))) if emb else 0
        return np.zeros((dim,), dtype=np.float32)

    dim = len(next(iter(emb.values())))
    x = np.zeros((dim,), dtype=np.float32)
    for a, p in d.items():
        v = emb.get(a)
        if v is None:
            continue
        x += float(p) * v.astype(np.float32, copy=False)
    return x


def _segment_vector_argmax_onehot(seg: Segment, *, alphabet: List[str], na_label: str) -> np.ndarray:
    d = _drop_na_and_renorm(seg.probs, na_label=na_label)
    x = np.zeros((len(alphabet),), dtype=np.float32)
    if not d:
        return x
    a = max(d.items(), key=lambda kv: kv[1])[0]
    try:
        x[alphabet.index(a)] = 1.0
    except ValueError:
        pass
    return x


def _segment_vector_weighted_onehot(seg: Segment, *, alphabet: List[str], na_label: str) -> np.ndarray:
    d = _drop_na_and_renorm(seg.probs, na_label=na_label)
    x = np.zeros((len(alphabet),), dtype=np.float32)
    for a, p in d.items():
        try:
            x[alphabet.index(a)] = float(p)
        except ValueError:
            continue
    return x


def _make_supervised_dataset(
    cases: Dict[int, CaseData],
    *,
    alphabet: List[str],  # excluding NA
    na_label: str,
    max_len: int,
    representation: str,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) for next-activity prediction.

    X: (N, max_len, D)
    y: (N,) class indices in [0, |alphabet|-1]
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    # Determine input dimension D
    if representation == "expected_embedding":
        if embeddings is None:
            raise ValueError("embeddings required for expected_embedding representation")
        D = len(next(iter(embeddings.values()))) if embeddings else 0
    elif representation in ("argmax_onehot", "weighted_onehot"):
        D = len(alphabet)
    else:
        raise ValueError(f"Unknown representation: {representation!r}")

    for _cid, case in cases.items():
        segs = case.segments
        # precompute per-segment vectors for this trace
        vecs: List[np.ndarray] = []
        for seg in segs:
            if representation == "expected_embedding":
                v = _segment_vector_expected_embedding(seg, emb=embeddings or {}, alphabet=alphabet, na_label=na_label)
            elif representation == "argmax_onehot":
                v = _segment_vector_argmax_onehot(seg, alphabet=alphabet, na_label=na_label)
            else:
                v = _segment_vector_weighted_onehot(seg, alphabet=alphabet, na_label=na_label)
            vecs.append(v)

        for i, seg in enumerate(segs):
            y_lab = _first_non_na_after_end(case, end_t=seg.end_t, na_label=na_label)
            if y_lab is None:
                continue
            if y_lab == na_label:
                continue
            try:
                y = alphabet.index(y_lab)
            except ValueError:
                continue

            prefix = vecs[: i + 1]
            if not prefix:
                continue
            if len(prefix) > max_len:
                prefix = prefix[-max_len:]
            pad = max_len - len(prefix)
            x = np.zeros((max_len, D), dtype=np.float32)
            for j, vv in enumerate(prefix):
                x[pad + j, :] = vv
            X_list.append(x)
            y_list.append(int(y))

    if not X_list:
        return np.zeros((0, max_len, D), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(X_list, axis=0), np.asarray(y_list, dtype=np.int64)


# -----------------------------------------------------------------------------
# Train/eval model (Evermann-style LSTM)
# -----------------------------------------------------------------------------


def _train_eval_evermann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    num_classes: int,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Train a single-layer LSTM classifier and return accuracy on test.
    """
    # Local import so the rest of the repo can be used without TF if needed.
    import tensorflow as tf

    # Best-effort determinism for repeated runs (still may vary slightly on GPU/MPS).
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)
    try:
        # TF 2.13+: enable deterministic kernels when available.
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass
    np.random.seed(seed)
    random.seed(seed)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=2,
        callbacks=cb,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0, batch_size=int(batch_size))
    return {"test_acc": float(test_acc), "test_loss": float(test_loss)}


# -----------------------------------------------------------------------------
# Public entry point (PyCharm-friendly)
# -----------------------------------------------------------------------------


def run_uncertain_next_activity_prediction(
    *,
    model_id: str,
    embedding_method: str,
    window_size: int,
    representation: str,
    seed: int = 42,
    split_train: float = 0.64,
    split_val: float = 0.16,
    split_strategy: str = "shuffle_seeded",  # "shuffle_seeded" | "sorted"
    max_len: int = 20,
    top_k_event: int = 3,
    na_label: str = "NA",
    embedding_training: str = "top3_uncertain",  # "top3_uncertain" | "top1_determinized"
    epochs: int = 50,
    batch_size: int = 64,
    # Optional override to make the benchmark OS/environment-independent.
    # If set, must be a directory containing `uncertain_event_data/`.
    data_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Run one configuration on IKEA ASM split=test for a single model_id.

    representation:
      - "expected_embedding"
      - "argmax_onehot"
      - "weighted_onehot"
    """
    # Load cases
    cases_all = load_ikea_split_test_model(
        model_id=model_id,
        data_root=data_root,
        na_label=na_label,
        top_k_event=int(top_k_event),
    )
    case_ids = sorted(cases_all.keys())
    if split_strategy not in ("shuffle_seeded", "sorted"):
        raise ValueError("split_strategy must be 'shuffle_seeded' or 'sorted'")
    if split_strategy == "shuffle_seeded":
        rnd = random.Random(int(seed))
        rnd.shuffle(case_ids)

    n = len(case_ids)
    n_train = int(round(split_train * n))
    n_val = int(round(split_val * n))
    n_train = max(1, min(n - 2, n_train))
    n_val = max(1, min(n - n_train - 1, n_val))
    train_ids = set(case_ids[:n_train])
    val_ids = set(case_ids[n_train : n_train + n_val])
    test_ids = set(case_ids[n_train + n_val :])

    cases_train = {cid: cases_all[cid] for cid in train_ids}
    cases_val = {cid: cases_all[cid] for cid in val_ids}
    cases_test = {cid: cases_all[cid] for cid in test_ids}

    # Alphabet: infer from GT labels in this dataset, excluding NA.
    # (Keeps the benchmark self-contained and matches IKEA label schema.)
    alphabet_set = set()
    for c in cases_all.values():
        if c.gt_label is not None:
            for lab in c.gt_label:
                if lab != na_label:
                    alphabet_set.add(lab)
        elif c.gt_segments is not None:
            for (_s, _e, lab) in c.gt_segments:
                if lab != na_label:
                    alphabet_set.add(lab)
    alphabet = sorted(alphabet_set)

    # Embeddings (trained on train+val), only needed for expected_embedding.
    emb: Optional[Dict[str, np.ndarray]] = None
    emb_dim = None
    if representation == "expected_embedding":
        if embedding_training not in ("top3_uncertain", "top1_determinized"):
            raise ValueError("embedding_training must be 'top3_uncertain' or 'top1_determinized'")
        log_train_val = build_uncertain_event_log_from_cases(
            {**cases_train, **cases_val},
            na_label=na_label,
            determinize_top1=(embedding_training == "top1_determinized"),
        )
        emb = _compute_uncertain_embeddings(
            log_train_val,
            embedding_method=embedding_method,
            window_size=int(window_size),
            na_label=na_label,
        )
        # ensure all alphabet labels exist (missing -> zeros)
        emb_dim = len(next(iter(emb.values()))) if emb else 0
        for a in alphabet:
            if a not in emb:
                emb[a] = np.zeros((emb_dim,), dtype=np.float32)

    # Build datasets
    X_train, y_train = _make_supervised_dataset(
        cases_train,
        alphabet=alphabet,
        na_label=na_label,
        max_len=int(max_len),
        representation=representation,
        embeddings=emb,
    )
    X_val, y_val = _make_supervised_dataset(
        cases_val,
        alphabet=alphabet,
        na_label=na_label,
        max_len=int(max_len),
        representation=representation,
        embeddings=emb,
    )
    X_test, y_test = _make_supervised_dataset(
        cases_test,
        alphabet=alphabet,
        na_label=na_label,
        max_len=int(max_len),
        representation=representation,
        embeddings=emb,
    )

    metrics = _train_eval_evermann(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        num_classes=len(alphabet),
        seed=int(seed),
        epochs=int(epochs),
        batch_size=int(batch_size),
    )

    return {
        "model_id": model_id,
        "embedding_method": embedding_method,
        "window_size": int(window_size),
        "representation": representation,
        "seed": int(seed),
        "top_k_event": int(top_k_event),
        "na_label": str(na_label),
        "embedding_training": str(embedding_training),
        "max_len": int(max_len),
        "split_strategy": str(split_strategy),
        "data_root": str(Path(data_root).resolve()) if data_root is not None else str(Path(ROOT_DIR).resolve()),
        "n_cases": len(cases_all),
        "n_train_cases": len(cases_train),
        "n_val_cases": len(cases_val),
        "n_test_cases": len(cases_test),
        "n_train_samples": int(X_train.shape[0]),
        "n_val_samples": int(X_val.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        **metrics,
    }



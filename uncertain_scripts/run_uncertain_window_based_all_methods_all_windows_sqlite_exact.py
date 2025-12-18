"""
Exact, low-RAM runner for window-based uncertain count methods using SQLite aggregation.

This keeps results exact (no approximation), but avoids holding the massive
`expected_counts[a][ctx]` dicts in RAM.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import os
import sqlite3

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR
from uncertain_utils.uncertain_xes_reader import read_uncertain_xes
from distances.uncertain_activity_distances.data_util.uncertain_window_sqlite_backend import (
    SqliteCountsConfig,
    window_count_to_sqlite,
    compute_ac_distances_from_sqlite,
    compute_aa_from_sqlite,
)


# =============================================================================
# Hardcoded config (PyCharm-friendly)
# =============================================================================

XES_PATH = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"
WINDOW_SIZES = [3, 5, 9]
NA_LABEL = "NA"
PROB_THRESHOLD = 0.05

# Where to store temporary DBs (one per pass)
TMP_DIR = Path(ROOT_DIR) / "uncertain_tmp_sqlite"
DELETE_DB_AFTER_PASS = True

# Optional speed mode: keep SQLite DB in RAM if machine has enough memory.
SQLITE_IN_MEMORY_IF_ENOUGH_RAM = True
SQLITE_IN_MEMORY_MIN_TOTAL_RAM_GB = 50.0


def safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        pass


def total_ram_gb() -> float:
    # No extra deps: use sysconf where available.
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return (page * pages) / (1024**3)
    except Exception:
        return 0.0


if __name__ == "__main__":
    safe_print("Exact SQLite runner (window-based)")
    safe_print(f"xes={XES_PATH}")
    safe_print(f"window_sizes={WINDOW_SIZES} prob_threshold={PROB_THRESHOLD}")

    log = read_uncertain_xes(XES_PATH)

    # Build label compression map once (IDs reduce DB size and speed comparisons)
    all_labels = sorted(log.activities(exclude=set()))
    keep = {".", NA_LABEL}
    to_map = [a for a in all_labels if a not in keep]
    label_map = {a: str(i) for i, a in enumerate(to_map)}
    label_map["."] = "."
    label_map[NA_LABEL] = NA_LABEL
    inv_label_map = {v: k for k, v in label_map.items()}

    cfg = SqliteCountsConfig(prob_threshold=PROB_THRESHOLD, na_label=NA_LABEL, pad_token=".")

    results = {}
    ram_gb = total_ram_gb()
    use_mem = bool(SQLITE_IN_MEMORY_IF_ENOUGH_RAM and ram_gb >= SQLITE_IN_MEMORY_MIN_TOTAL_RAM_GB and ram_gb > 0.0)
    safe_print(f"sqlite_backend={'IN_MEMORY' if use_mem else 'ON_DISK'} total_ram_gb={ram_gb:.1f} threshold_gb={SQLITE_IN_MEMORY_MIN_TOTAL_RAM_GB:.1f}")
    if use_mem:
        safe_print("NOTE: in-memory SQLite is exact + fast, but can still OOM if the DB grows too large.")

    for ctx_kind_name, ctx_kind in [("Seq", "seq"), ("MSet", "mset")]:
        for w in WINDOW_SIZES:
            db_path = TMP_DIR / f"counts__{ctx_kind_name.lower()}__w{w}__thr{PROB_THRESHOLD:.2f}.sqlite"
            if (not use_mem) and db_path.exists():
                db_path.unlink()

            safe_print(f"\n[pass] start kind={ctx_kind_name} w={w} -> {':memory:' if use_mem else db_path.name}")
            t0 = time.time()

            if use_mem:
                conn = sqlite3.connect(":memory:")
                activity_freq = window_count_to_sqlite(
                    log,
                    window_size=w,
                    context_kind=ctx_kind,
                    label_map=label_map,
                    cfg=cfg,
                    conn=conn,
                    progress=lambda m: safe_print(m),
                )
            else:
                activity_freq = window_count_to_sqlite(
                    log,
                    window_size=w,
                    context_kind=ctx_kind,
                    label_map=label_map,
                    cfg=cfg,
                    db_path=db_path,
                    progress=lambda m: safe_print(m),
                )
                conn = sqlite3.connect(str(db_path))

            # AC
            ac_raw, ac_pmi, ac_ppmi = compute_ac_distances_from_sqlite(conn, activity_freq=activity_freq)
            # AA
            aa_raw, aa_pmi, aa_ppmi = compute_aa_from_sqlite(conn, activity_freq=activity_freq)
            conn.close()

            # map back to original labels
            def remap(dist):
                out = {}
                for (a, b), v in dist.items():
                    out[(inv_label_map.get(a, a), inv_label_map.get(b, b))] = v
                return out

            results[f"Uncertain-Window AC {ctx_kind_name} w_{w}"] = remap(ac_raw)
            results[f"Uncertain-Window AC {ctx_kind_name} PMI w_{w}"] = remap(ac_pmi)
            results[f"Uncertain-Window AC {ctx_kind_name} PPMI w_{w}"] = remap(ac_ppmi)
            results[f"Uncertain-Window AA {ctx_kind_name} w_{w}"] = remap(aa_raw)
            results[f"Uncertain-Window AA {ctx_kind_name} PMI w_{w}"] = remap(aa_pmi)
            results[f"Uncertain-Window AA {ctx_kind_name} PPMI w_{w}"] = remap(aa_ppmi)

            dt = time.time() - t0
            safe_print(f"[pass] done kind={ctx_kind_name} w={w} (dt={dt:.1f}s)")

            if (not use_mem) and DELETE_DB_AFTER_PASS:
                try:
                    db_path.unlink()
                except OSError:
                    pass

    safe_print(f"\nDone. Built {len(results)} distance matrices (exact, sqlite).")



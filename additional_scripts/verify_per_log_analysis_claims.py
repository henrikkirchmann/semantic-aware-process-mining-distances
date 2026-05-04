"""
verify_per_log_analysis_claims.py
----------------------------------
Reproduces and verifies every numerical claim made in the paper's
"Per-Log Analysis" discussion section from the committed result files.

Run from the repository root:
    python additional_scripts/verify_per_log_analysis_claims.py

Required files (both committed):
    results/activity_distances/intrinsic_summary/intrinsic_results_per_log_and_method.csv
    log_stats.csv
"""

import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_CSV   = os.path.join(ROOT, "results", "activity_distances",
                         "intrinsic_summary", "intrinsic_results_per_log_and_method.csv")
STATS_CSV = os.path.join(ROOT, "log_stats.csv")

res   = pd.read_csv(RES_CSV)
stats = pd.read_csv(STATS_CSV)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
errors = 0

def check(label, condition, detail=""):
    global errors
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    if detail:
        print(f"         {detail}")
    if not condition:
        errors += 1

# ---------------------------------------------------------------------------
# 1. Chiorrini avg I_comp
# ---------------------------------------------------------------------------
print("\n=== 1. Chiorrini avg I_comp (paper: 0.93) ===")
chi = res[res["method"] == "Chiorrini 2022 Embedding Process Structure"]
avg_comp = chi["I_comp"].mean()
check(f"avg I_comp = {avg_comp:.2f}", abs(avg_comp - 0.93) < 0.005,
      f"actual: {avg_comp:.4f}")

# ---------------------------------------------------------------------------
# 2. Chiorrini I_nn on BPIC15 (corrected range: 0.13–0.18)
# ---------------------------------------------------------------------------
print("\n=== 2. Chiorrini I_nn on BPIC15 family (corrected: 0.13–0.18) ===")
bpic15_chi = chi[chi["log_name"].str.startswith("BPIC15")][["log_name", "I_nn"]].sort_values("log_name")
print(bpic15_chi.to_string(index=False))
lo, hi = bpic15_chi["I_nn"].min(), bpic15_chi["I_nn"].max()
check(f"range {lo:.2f}–{hi:.2f} within 0.13–0.18",
      0.12 <= lo <= 0.14 and 0.17 <= hi <= 0.19,
      "(paper incorrectly stated 0.17–0.27; corrected values used here)")

# ---------------------------------------------------------------------------
# 3. Logs where Chiorrini wins I_nn
# ---------------------------------------------------------------------------
print("\n=== 3. Logs where Chiorrini wins I_nn (expected: BPIC12_O, BPIC12_W_Complete, CCC19) ===")
winners = res.loc[res.groupby("log_name")["I_nn"].idxmax(), ["log_name", "method", "I_nn"]]
chi_wins = winners[winners["method"].str.contains("Chiorrini")]["log_name"].tolist()
print(f"  Chiorrini wins on: {sorted(chi_wins)}")
expected_wins = {"BPIC12_O", "BPIC12_W_Complete", "CCC19"}
check("wins exactly on BPIC12_O, BPIC12_W_Complete, CCC19",
      set(chi_wins) == expected_wins,
      f"actual: {sorted(chi_wins)}")
check("does NOT win on any BPIC13 log",
      not any("BPIC13" in w for w in chi_wins),
      "(paper incorrectly included 'BPIC13 variants')")

# ---------------------------------------------------------------------------
# 4. Autoencoder never wins I_nn on any log
# ---------------------------------------------------------------------------
print("\n=== 4. Autoencoder never wins best I_nn on any log ===")
auto = res[res["method"].str.contains("Gamallo")]
auto_wins = []
for log, g in res.groupby("log_name"):
    v = auto[auto["log_name"] == log]["I_nn"].values
    if len(v) and v[0] == g["I_nn"].max():
        auto_wins.append(log)
check("autoencoder wins 0 logs", len(auto_wins) == 0,
      f"won on: {auto_wins}" if auto_wins else "confirmed: no wins")

# ---------------------------------------------------------------------------
# 5. Correlation: count-based I_nn vs log properties
# ---------------------------------------------------------------------------
print("\n=== 5. Pearson correlations (count-based avg I_nn vs log properties) ===")
cb = (res[res["method"].str.startswith("Activity-")]
        .groupby("log_name")["I_nn"].mean()
        .reset_index())
cb = cb.merge(stats[["log_name", "ratio_trace_variants", "num_traces", "avg_trace_length"]],
              on="log_name")
corr = cb[["I_nn", "ratio_trace_variants", "num_traces", "avg_trace_length"]].corr()
r_ratio  = corr.loc["I_nn", "ratio_trace_variants"]
r_traces = corr.loc["I_nn", "num_traces"]
r_length = corr.loc["I_nn", "avg_trace_length"]
print(f"  I_nn vs ratio_trace_variants: {r_ratio:.3f}")
print(f"  I_nn vs num_traces:           {r_traces:.3f}")
print(f"  I_nn vs avg_trace_length:     {r_length:.3f}")
check("trace-variant ratio negatively correlated (r < -0.3)", r_ratio < -0.3)
check("num_traces positively correlated  (r > 0.2)",          r_traces > 0.2)
check("avg_trace_length negatively correlated (r < -0.3)",    r_length < -0.3)

# ---------------------------------------------------------------------------
# 6. BPIC15 top-3: Activity-Context BoW/N-Gram dominate; act2vec on BPIC15_5
# ---------------------------------------------------------------------------
print("\n=== 6. BPIC15 I_nn top-3 per log ===")
for log in ["BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]:
    top3 = res[res["log_name"] == log].nlargest(3, "I_nn")[["method", "I_nn"]]
    print(f"\n  {log}:")
    for _, r in top3.iterrows():
        print(f"    {r['method'][:65]:65s}  I_nn={r['I_nn']:.3f}")
    top_methods = top3["method"].tolist()
    all_activity_context = all("Activity-Context" in m for m in top_methods)
    has_act2vec = any("act2vec" in m for m in top_methods)
    if log == "BPIC15_5":
        check(f"{log}: act2vec or Activity-Context leads", all_activity_context or has_act2vec)
    else:
        check(f"{log}: Activity-Context variants in top-3", all_activity_context or
              sum("Activity-Context" in m for m in top_methods) >= 2)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
if errors == 0:
    print(f"  All checks passed.")
else:
    print(f"  {errors} check(s) failed — see details above.")
    sys.exit(1)

"""
Create a new uncertain XES where tiny probabilities are removed and the remainder renormalized.

PyCharm-friendly: edit the constants below and run.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on PYTHONPATH when running directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from definitions import ROOT_DIR
from uncertain_utils.uncertain_xes_probability_thresholding import (
    ThresholdingConfig,
    threshold_xes_probs_json,
)


# =============================================================================
# Hardcoded config
# =============================================================================

INPUT_XES = Path(ROOT_DIR) / "xes_uncertain_gt__no_na.xes"
OUTPUT_XES = Path(ROOT_DIR) / "uncertain_event_logs" / "xes_uncertain_gt__no_na__p_ge_0_01.xes"

# Threshold: set all probs < 1% to zero (remove key), then renormalize remaining mass.
THRESHOLD = 0.01

# Optional: also remove NA and renormalize (leave empty set if you want to keep NA as an option)
DROP_LABELS = set()  # e.g., {"NA"}

# If you have a label->id mapping, you can load it and pass into `label_name_to_id`.
# Otherwise we update `pred:label` only and leave `pred:label_id` unchanged.
LABEL_NAME_TO_ID = None  # dict[str,int] | None


if __name__ == "__main__":
    cfg = ThresholdingConfig(
        probs_key="probs_json",
        threshold=THRESHOLD,
        drop_labels=frozenset(DROP_LABELS),
        renormalize=True,
    )

    print(f"Input : {INPUT_XES}")
    print(f"Output: {OUTPUT_XES}")
    print(f"Threshold: p < {THRESHOLD} -> removed; renormalize remainder; drop_labels={sorted(DROP_LABELS)}")

    events_seen, events_updated = threshold_xes_probs_json(
        input_xes=INPUT_XES,
        output_xes=OUTPUT_XES,
        config=cfg,
        label_name_to_id=LABEL_NAME_TO_ID,
    )

    print(f"Done. events_seen={events_seen} events_updated={events_updated}")





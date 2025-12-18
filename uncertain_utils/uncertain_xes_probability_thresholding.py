"""
Threshold & renormalize `probs_json` in uncertain XES logs.

Motivation
----------
Some uncertain logs (e.g., IKEA ASM exports) store *dense* probability distributions
per event: many activities have very small non-zero probability. Enumerating all
trace realizations becomes infeasible because per-event support sizes are large.

To mitigate this, we can *edit the log* by:
1) Setting probabilities below a threshold (e.g., 1%) to zero (i.e., removing keys)
2) Renormalizing the remaining distribution to sum to 1 again

This mirrors the idea used in the IKEA repo when removing `NA` and renormalizing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import xml.etree.ElementTree as ET


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


@dataclass(frozen=True)
class ThresholdingConfig:
    probs_key: str = "probs_json"
    threshold: float = 0.01  # 1%
    drop_labels: Set[str] = frozenset()
    renormalize: bool = True

    # XES keys to update (optional)
    pred_label_key: str = "pred:label"
    pred_label_id_key: str = "pred:label_id"


def threshold_and_renormalize_distribution(
    probs: Dict[str, float],
    *,
    threshold: float,
    drop_labels: Set[str],
    renormalize: bool,
) -> Dict[str, float]:
    """
    Remove labels in `drop_labels` and all labels with p < threshold, then renormalize.

    If all labels are removed, we keep the original argmax label and set it to 1.0
    (so every event remains well-defined).
    """
    if not probs:
        return {}

    # Clean numeric + drop
    cleaned = {}
    for a, p in probs.items():
        if a in drop_labels:
            continue
        try:
            pf = float(p)
        except Exception:
            continue
        if pf < threshold or pf <= 0.0:
            continue
        cleaned[a] = pf

    if not cleaned:
        # Fallback: keep original argmax (even if below threshold)
        a_max = max(probs.items(), key=lambda kv: float(kv[1]))[0]
        return {str(a_max): 1.0}

    if renormalize:
        s = sum(cleaned.values())
        if s > 0:
            cleaned = {a: p / s for a, p in cleaned.items()}
    return cleaned


def _find_event_kv_elems(event_elem: ET.Element) -> Dict[str, ET.Element]:
    """
    Map key -> child element for XES primitive children (string/int/date/float/boolean).
    """
    out = {}
    for child in list(event_elem):
        if _local_name(child.tag) in {"string", "int", "date", "float", "boolean"}:
            k = child.attrib.get("key")
            if k:
                out[k] = child
    return out


def threshold_xes_probs_json(
    *,
    input_xes: str | Path,
    output_xes: str | Path,
    config: ThresholdingConfig,
    label_name_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[int, int]:
    """
    Read XES, threshold+renormalize `probs_json` for each event, update `pred:label`
    (and `pred:label_id` if mapping provided), then write XES to `output_xes`.

    Returns
    -------
    (events_seen, events_updated)
    """
    in_path = Path(input_xes)
    out_path = Path(output_xes)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    tree = ET.parse(str(in_path))
    root = tree.getroot()

    events_seen = 0
    events_updated = 0

    for elem in root.iter():
        if _local_name(elem.tag) != "event":
            continue
        events_seen += 1
        kv = _find_event_kv_elems(elem)

        probs_elem = kv.get(config.probs_key)
        if probs_elem is None:
            continue
        raw = probs_elem.attrib.get("value")
        if not raw:
            continue

        try:
            probs = json.loads(raw)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON in {config.probs_key}: {raw!r}") from e

        new_probs = threshold_and_renormalize_distribution(
            probs,
            threshold=config.threshold,
            drop_labels=set(config.drop_labels),
            renormalize=config.renormalize,
        )

        # Write back probs_json
        probs_elem.attrib["value"] = json.dumps(new_probs, ensure_ascii=False, separators=(",", ":"))
        events_updated += 1

        # Update pred:label (string) if present
        if config.pred_label_key in kv:
            pred_label = max(new_probs.items(), key=lambda kv2: kv2[1])[0]
            kv[config.pred_label_key].attrib["value"] = str(pred_label)

        # Update pred:label_id (int) if present and mapping provided
        if label_name_to_id is not None and config.pred_label_id_key in kv:
            pred_label = max(new_probs.items(), key=lambda kv2: kv2[1])[0]
            if pred_label in label_name_to_id:
                kv[config.pred_label_id_key].attrib["value"] = str(int(label_name_to_id[pred_label]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=False)
    return events_seen, events_updated





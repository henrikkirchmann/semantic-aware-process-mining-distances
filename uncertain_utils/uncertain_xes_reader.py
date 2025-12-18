"""
Uncertain XES reader (stdlib-only, streaming).

Why this exists
---------------
The original codebase reads deterministic XES logs via pm4py and then converts them
into a list-of-traces-of-activities. For uncertain event data, each event additionally
stores a probability distribution over activities, typically as an XES string attribute
named `probs_json` (see the IKEA ASM uncertain logs repo referenced in your prompt).

This module parses such an XES and builds an in-memory structure that is convenient
for later algorithms:

    UncertainEventLog
      - traces: list[UncertainTrace]
          - events: list[UncertainEvent]
              - activity_probs: dict[str, float]  # probability distribution over activities

Notes
-----
- This parser uses `xml.etree.ElementTree.iterparse` to avoid loading the whole file
  at once (important because many XES files are huge and can be formatted as a single line).
- If an event does *not* have a `probs_json` attribute, we treat it as deterministic by
  assigning probability 1.0 to `concept:name`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET


def _local_name(tag: str) -> str:
    """Strip XES XML namespaces, returning the local element name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


@dataclass(frozen=True)
class UncertainEvent:
    """
    One event with an (optional) activity probability distribution.

    `attributes` retains all parsed XES event attributes (ints, strings, dates as strings).
    """

    activity_probs: Dict[str, float]
    attributes: Dict[str, Any] = field(default_factory=dict)

    def most_likely_activity(self) -> str:
        if not self.activity_probs:
            return "__EMPTY__"
        return max(self.activity_probs.items(), key=lambda kv: kv[1])[0]


@dataclass(frozen=True)
class UncertainTrace:
    """
    One trace (case).
    """

    trace_id: str
    case_name: str
    events: List[UncertainEvent]
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UncertainEventLog:
    """
    Whole event log.
    """

    traces: List[UncertainTrace]
    log_name: Optional[str] = None

    def iter_events(self) -> Iterator[UncertainEvent]:
        for tr in self.traces:
            yield from tr.events

    def activities(self, exclude: Optional[Set[str]] = None) -> Set[str]:
        ex = exclude or set()
        acts: Set[str] = set()
        for ev in self.iter_events():
            for a in ev.activity_probs.keys():
                if a not in ex:
                    acts.add(a)
        return acts


def _parse_kv_element(elem: ET.Element) -> Tuple[str, Any]:
    """
    Parse a single XES attribute element like:
      <string key="concept:name" value="..." />
      <int key="segment:start_timestamp" value="58" />
      <date key="time:timestamp" value="1970-01-01T00:00:58Z" />

    Returns (key, value) with ints converted to int, floats to float where applicable.
    """
    key = elem.attrib.get("key")
    val = elem.attrib.get("value")
    if key is None:
        return ("__INVALID__", None)

    kind = _local_name(elem.tag)
    if kind == "int":
        try:
            return (key, int(val))  # type: ignore[arg-type]
        except Exception:
            return (key, val)
    if kind == "float":
        try:
            return (key, float(val))  # type: ignore[arg-type]
        except Exception:
            return (key, val)
    # date, string, boolean, etc. -> keep as string for now
    return (key, val)


def _clean_and_normalize_probs(
    probs: Dict[str, float],
    *,
    drop_activities: Optional[Set[str]] = None,
    min_prob: float = 0.0,
    renormalize: bool = True,
) -> Dict[str, float]:
    drop = drop_activities or set()
    cleaned: Dict[str, float] = {}
    for a, p in probs.items():
        if a in drop:
            continue
        if p is None:
            continue
        try:
            pf = float(p)
        except Exception:
            continue
        if pf <= 0.0 or pf < min_prob:
            continue
        cleaned[a] = pf

    if not cleaned:
        return {}

    if renormalize:
        s = sum(cleaned.values())
        if s > 0:
            cleaned = {a: p / s for a, p in cleaned.items()}
    return cleaned


def read_uncertain_xes(
    xes_path: str | Path,
    *,
    probs_key: str = "probs_json",
    activity_key: str = "concept:name",
    trace_id_key: str = "concept:name",
    case_name_key: str = "case:name",
    drop_activities: Optional[Set[str]] = None,
    min_prob: float = 0.0,
    renormalize: bool = True,
    limit_traces: Optional[int] = None,
) -> UncertainEventLog:
    """
    Read an uncertain XES into an `UncertainEventLog`.

    Hardcoded keys are aligned with the IKEA ASM uncertain XES export described in:
    - https://github.com/henrikkirchmann/IKEA_ASM_UncertainEventLogs

    Parameters
    ----------
    probs_key:
        XES event attribute that stores a JSON dict activity->probability. If absent,
        the event is treated as deterministic using `activity_key`.
    drop_activities:
        Activities to remove from the probability distribution (e.g., {"NA"}).
    min_prob:
        Drop activities with probability below this threshold (useful for dense distributions).
    renormalize:
        If True, renormalize probabilities after dropping activities.
    limit_traces:
        For quick debugging, stop after reading N traces.
    """

    path = Path(xes_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    traces: List[UncertainTrace] = []
    log_name: Optional[str] = None

    current_trace_attrs: Dict[str, Any] = {}
    current_trace_events: List[UncertainEvent] = []
    current_trace_id: str = ""
    current_case_name: str = ""
    in_trace = False
    in_event = False

    # Parse end-events so we can "finalize" traces/events and clear elements to free memory.
    for event, elem in ET.iterparse(str(path), events=("start", "end")):
        tag = _local_name(elem.tag)

        if event == "start" and tag == "trace":
            in_trace = True
            current_trace_attrs = {}
            current_trace_events = []
            current_trace_id = ""
            current_case_name = ""

        if event == "start" and tag == "event":
            in_event = True

        if event == "end" and tag == "trace":
            # finalize trace
            tr = UncertainTrace(
                trace_id=current_trace_id,
                case_name=current_case_name,
                events=current_trace_events,
                attributes=current_trace_attrs,
            )
            traces.append(tr)
            # memory
            elem.clear()
            in_trace = False
            if limit_traces is not None and len(traces) >= limit_traces:
                break

        if event == "end" and tag == "event":
            attrs: Dict[str, Any] = {}
            for child in list(elem):
                k, v = _parse_kv_element(child)
                if k != "__INVALID__":
                    attrs[k] = v

            # Determine activity probability distribution.
            if probs_key in attrs and attrs[probs_key]:
                raw = attrs[probs_key]
                try:
                    probs_dict = json.loads(raw)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSON in '{probs_key}' for event: {raw!r}") from e
                # Ensure float values
                probs = {str(a): float(p) for a, p in probs_dict.items()}
            else:
                # Deterministic fallback
                a = attrs.get(activity_key, "__MISSING__")
                probs = {str(a): 1.0}

            probs = _clean_and_normalize_probs(
                probs, drop_activities=drop_activities, min_prob=min_prob, renormalize=renormalize
            )

            current_trace_events.append(UncertainEvent(activity_probs=probs, attributes=attrs))
            elem.clear()
            in_event = False

        # Collect log-level and trace-level attributes.
        if event == "end" and tag in ("string", "int", "date", "float"):
            k, v = _parse_kv_element(elem)
            if k == "__INVALID__":
                continue

            # log-level name: first `<string key="concept:name" ...>` under <log>.
            # Note: Many XES logs include `<global ...><string key="concept:name" value="__INVALID__"/></global>`
            # before the actual log-level name. We therefore ignore "__INVALID__" and allow later values to
            # overwrite it.
            if not in_trace and k == activity_key and v is not None:
                vv = str(v)
                if vv != "__INVALID__" and (log_name is None or log_name == "__INVALID__"):
                    log_name = vv

            # trace-level attrs (avoid mixing in event attrs)
            if in_trace and not in_event:
                current_trace_attrs[k] = v
                if k == trace_id_key and not current_trace_id:
                    current_trace_id = str(v)
                if k == case_name_key and not current_case_name:
                    current_case_name = str(v)

    return UncertainEventLog(traces=traces, log_name=log_name)



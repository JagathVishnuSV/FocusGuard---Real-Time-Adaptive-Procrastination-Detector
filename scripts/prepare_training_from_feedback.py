#!/usr/bin/env python3
"""Prepare a training CSV from web feedback and feature snapshots.

This script handles two cases:
 - feedback.jsonl entries already include a `features` map -> directly use them
 - feedback.jsonl only contains labels -> join to `feature_snapshots.jsonl` by
   session_id + nearest timestamp within a tolerance (default 5s)

Outputs:
 - data/personalization/feedback_for_training.csv  (labeled rows only)
 - data/personalization/snapshots_with_labels.csv  (all snapshots with any matched label)

The script uses mean imputation for missing feature values computed from the
complete snapshot corpus.
"""
from pathlib import Path
import json
import csv
import sys
from bisect import bisect_left
from typing import Dict, List, Any, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "personalization"
FEEDBACK = DATA_DIR / "feedback.jsonl"
SNAPSHOTS = DATA_DIR / "feature_snapshots.jsonl"
OUT = DATA_DIR / "feedback_for_training.csv"
OUT_ALL = DATA_DIR / "snapshots_with_labels.csv"

# tolerance in seconds to match a label to a snapshot
DEFAULT_TOLERANCE_S = 5.0

try:
    import config
except Exception:
    sys.path.insert(0, str(ROOT))
    import config  # type: ignore


def load_snapshots() -> Dict[str, List[Tuple[float, Dict[str, Any]]]]:
    """Return index: session_id -> sorted list of (timestamp, features)"""
    idx: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
    if not SNAPSHOTS.exists():
        return idx
    with SNAPSHOTS.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ts = float(entry.get("timestamp") or 0.0)
            sid = entry.get("session_id") or ""
            features = entry.get("features") or {}
            idx.setdefault(sid, []).append((ts, features))
    # sort per-session
    for sid in idx:
        idx[sid].sort(key=lambda x: x[0])
    return idx


def compute_feature_means(snapshots_index: Dict[str, List[Tuple[float, Dict[str, Any]]]]) -> Dict[str, float]:
    sums: Dict[str, float] = {k: 0.0 for k in config.FEATURE_NAMES}
    counts: Dict[str, int] = {k: 0 for k in config.FEATURE_NAMES}
    for sid, snaps in snapshots_index.items():
        for ts, feats in snaps:
            for k in config.FEATURE_NAMES:
                v = None
                if isinstance(feats, dict):
                    v = feats.get(k)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                sums[k] += fv
                counts[k] += 1
    means: Dict[str, float] = {}
    for k in config.FEATURE_NAMES:
        if counts[k] > 0:
            means[k] = sums[k] / counts[k]
        else:
            means[k] = 0.0
    return means


def load_label_index(path: Path) -> Dict[str, List[Tuple[Optional[float], int, Dict[str, Any]]]]:
    """Load JSONL of labels into index: session_id -> sorted list of (timestamp, label_int, raw_entry)"""
    idx: Dict[str, List[Tuple[Optional[float], int, Dict[str, Any]]]] = {}
    if not path.exists():
        return idx
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            sid = entry.get("session_id") or ""
            ts_val = entry.get("timestamp")
            try:
                ts = float(ts_val) if ts_val is not None else None
            except Exception:
                ts = None
            # possible label fields
            label = entry.get("user_label") or entry.get("label")
            lbl: Optional[int] = None
            if isinstance(label, str):
                lstr = label.strip().lower()
                if lstr in {"distracted", "1", "true", "yes"}:
                    lbl = 1
                else:
                    try:
                        lbl = int(float(lstr))
                    except Exception:
                        lbl = 0
            elif isinstance(label, (int, float)):
                lbl = int(label)
            if lbl is None:
                continue
            idx.setdefault(sid, []).append((ts, lbl, entry))
    # sort per-session
    for sid in idx:
        idx[sid].sort(key=lambda x: (x[0] if x[0] is not None else 0.0))
    return idx


def find_nearest(sorted_list: List[Tuple[Optional[float], int, Dict[str, Any]]],
                 ts: float,
                 tolerance: float = DEFAULT_TOLERANCE_S) -> Optional[Tuple[Optional[float], int, Dict[str, Any]]]:
    """Find nearest (timestamp,label,entry) in sorted_list to ts within tolerance seconds.

    sorted_list elements have timestamp possibly None; treat None as far away.
    Returns the tuple or None.
    """
    if not sorted_list:
        return None
    # build an array of timestamps (use 0.0 for None to allow bisect)
    times = [(t if t is not None else 0.0) for (t, _, _) in sorted_list]
    pos = bisect_left(times, ts)
    candidates = []
    for idx in (pos - 1, pos, pos + 1):
        if 0 <= idx < len(sorted_list):
            cand = sorted_list[idx]
            c_ts = cand[0]
            if c_ts is None:
                continue
            if abs(c_ts - ts) <= tolerance:
                candidates.append((abs(c_ts - ts), cand))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def write_labeled_csv(path: Path, rows: List[Dict[str, Any]], feature_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No labeled rows to write to {path}")
        return
    fieldnames = list(feature_names) + ["label"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(out)
    print(f"Wrote {len(rows)} labeled rows to {path}")


def write_all_snapshots_csv(path: Path, rows: List[Dict[str, Any]], feature_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No snapshot rows to write to {path}")
        return
    fieldnames = list(feature_names) + ["label", "label_source", "session_id", "timestamp"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(out)
    print(f"Wrote {len(rows)} snapshot rows (with labels) to {path}")


def main(tolerance: float = DEFAULT_TOLERANCE_S) -> int:
    snapshots_index = load_snapshots()
    if not snapshots_index:
        print("No snapshots found; nothing to prepare.")
        return 1
    feature_means = compute_feature_means(snapshots_index)

    feedback_index = load_label_index(FEEDBACK)
    passive_index = load_label_index(DATA_DIR / "passive_labels.jsonl")

    labeled_rows: List[Dict[str, Any]] = []
    all_snapshot_rows: List[Dict[str, Any]] = []

    for sid, snaps in snapshots_index.items():
        for ts, feats in snaps:
            filled: Dict[str, Any] = {}
            for name in config.FEATURE_NAMES:
                val = None
                if isinstance(feats, dict):
                    val = feats.get(name)
                if val is None:
                    fv = feature_means.get(name, 0.0)
                else:
                    try:
                        fv = float(val)
                    except Exception:
                        fv = feature_means.get(name, 0.0)
                filled[name] = fv

            label = None
            label_source = ""
            # try user feedback first
            if sid in feedback_index:
                cand = find_nearest(feedback_index[sid], ts, tolerance=tolerance)
                if cand:
                    label = int(cand[1])
                    label_source = "user"
            # otherwise try passive
            if label is None and sid in passive_index:
                cand = find_nearest(passive_index[sid], ts, tolerance=tolerance)
                if cand:
                    label = int(cand[1])
                    label_source = "passive"

            row = dict(filled)
            row["label"] = label if label is not None else ""
            row["label_source"] = label_source
            row["session_id"] = sid
            row["timestamp"] = ts
            all_snapshot_rows.append(row)

            if label is not None:
                lr = dict(filled)
                lr["label"] = int(label)
                labeled_rows.append(lr)

    # write outputs
    write_labeled_csv(OUT, labeled_rows, config.FEATURE_NAMES)
    write_all_snapshots_csv(OUT_ALL, all_snapshot_rows, config.FEATURE_NAMES)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

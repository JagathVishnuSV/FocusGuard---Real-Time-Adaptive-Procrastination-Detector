#!/usr/bin/env python3
"""Convert user JSONL feedback to a training CSV that the training script can consume.

Writes to data/personalization/feedback_for_training.csv and includes the label column.
"""
from pathlib import Path
import json
import csv
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT = DATA_DIR / "personalization" / "feedback.jsonl"
OUTPUT = DATA_DIR / "personalization" / "feedback_for_training.csv"

try:
    import config
except Exception:
    # allow running from repo root where config is importable
    sys.path.insert(0, str(ROOT))
    import config

def main():
    if not INPUT.exists():
        print(f"No feedback file found at {INPUT}")
        return 1

    rows = []
    with INPUT.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            features = entry.get("features") or {}
            label = entry.get("user_label") or entry.get("label")
            if isinstance(label, str):
                label = 1 if label.lower() == "distracted" else 0
            elif isinstance(label, (int, float)):
                label = int(label)
            else:
                # Skip entries without label
                continue

            row = {name: float(features.get(name, 0.0)) for name in config.FEATURE_NAMES}
            row["label"] = label
            rows.append(row)

    if not rows:
        print("No usable feedback rows with features/labels found")
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=config.FEATURE_NAMES + ["label"]) 
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

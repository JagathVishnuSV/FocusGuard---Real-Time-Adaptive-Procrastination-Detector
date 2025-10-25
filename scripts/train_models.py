"""Utility script to train FocusGuard models from a prepared dataset.

Usage
-----
python scripts/train_models.py --dataset path/to/dataset.csv [--label-column label]

The dataset must include all feature columns listed in ``config.FEATURE_NAMES``.
If the label column is provided (defaults to ``label``), the random forest
classifier will be trained as well. Otherwise only the anomaly detector is
updated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ml import FocusGuardEnsemble
from ml.artifacts import ModelArtifact, read_metadata, write_metadata


DEFAULT_DATASET_PATH = config.DATA_DIR / "focusguard_windows_sessions.csv"


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    missing_features: List[str] = [
        feature for feature in config.FEATURE_NAMES if feature not in df.columns
    ]
    if missing_features:
        raise ValueError(
            "Dataset is missing required feature columns: " + ", ".join(missing_features)
        )
    return df


def register_artifact(artifact: ModelArtifact) -> None:
    """Upsert the artifact in the global registry."""
    registry_path = config.MODEL_REGISTRY_FILE
    existing = read_metadata(registry_path)
    updated = [entry for entry in existing if entry.name != artifact.name]
    updated.append(artifact)
    write_metadata(updated, registry_path)


def train_anomaly_detector(df: pd.DataFrame, ensemble: FocusGuardEnsemble) -> ModelArtifact:
    features = df[config.FEATURE_NAMES].to_numpy()
    report = ensemble.anomaly_pipeline.fit(features, config.MODEL_FILE)
    register_artifact(report.artifact)
    return report.artifact


def train_classifier(
    df: pd.DataFrame,
    ensemble: FocusGuardEnsemble,
    label_column: str,
) -> ModelArtifact:
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")

    labels = df[label_column].astype(int).to_numpy()
    features = df[config.FEATURE_NAMES].to_numpy()

    report = ensemble.classification_pipeline.fit(
        features,
        labels,
        config.RANDOM_FOREST_MODEL_FILE,
        config.SCALER_FILE,
    )
    ensemble.use_classifier = True
    register_artifact(report.artifact)
    return report.artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FocusGuard models from CSV data")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Path to the CSV dataset containing feature columns. "
            "Defaults to data/focusguard_windows_sessions.csv"
        ),
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the supervised label column (1=distraction, 0=focus)",
    )
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Only train the anomaly detector even if labels are present",
    )
    args = parser.parse_args()

    dataset_path: Path = args.dataset
    df = load_dataset(dataset_path)
    print(f"Loaded dataset with {len(df):,} rows from {dataset_path}")

    ensemble = FocusGuardEnsemble(config)

    anomaly_artifact = train_anomaly_detector(df, ensemble)
    print(f"Isolation Forest saved to: {anomaly_artifact.path}")

    classifier_artifact = None
    if not args.skip_classifier and args.label_column in df.columns:
        classifier_artifact = train_classifier(df, ensemble, args.label_column)
        print(f"Random Forest saved to: {classifier_artifact.path}")
    else:
        if args.skip_classifier:
            print("Classifier training skipped by flag")
        else:
            print(
                "Label column not found; classifier training skipped. "
                "Provide --label-column if your data contains labels."
            )

    # Persist any additional ensemble state (e.g., scaler) to the models directory.
    ensemble.save(config.MODELS_DIR)
    print(f"Artifacts persisted to {config.MODELS_DIR}")

    print("\nTraining complete. You can now restart the FocusGuard backend to load the new models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

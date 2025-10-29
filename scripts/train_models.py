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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ml import FocusGuardEnsemble
from ml.models.anomaly import AnomalyDetector
from ml.artifacts import ModelArtifact, read_metadata, write_metadata


DEFAULT_DATASET_PATH = config.DATA_DIR / "focusguard_windows_sessions_20000.csv"


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


def train_anomaly_detector(
    df: pd.DataFrame,
    ensemble: FocusGuardEnsemble,
    params_override: Optional[Dict[str, float]] = None,
) -> ModelArtifact:
    features = df[config.FEATURE_NAMES].to_numpy()
    report = ensemble.anomaly_pipeline.fit(
        features,
        config.MODEL_FILE,
        params_override=params_override,
    )
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
        config.CLASSIFIER_CALIBRATOR_FILE,
    )
    ensemble.use_classifier = True
    register_artifact(report.artifact)
    return report.artifact


def maybe_tune_isolation_forest(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    label_column: str,
) -> Optional[float]:
    if val_df is None or val_df.empty or label_column not in val_df.columns:
        return None

    val_labels = val_df[label_column].astype(int).to_numpy()
    if len(np.unique(val_labels)) < 2:
        # Cannot compute ROC-AUC with a single class present
        return None

    candidate_contamination = [0.03, 0.05, 0.08, 0.1]
    features_train = train_df[config.FEATURE_NAMES].to_numpy()
    features_val = val_df[config.FEATURE_NAMES].to_numpy()

    best_contamination: Optional[float] = None
    best_auc = -np.inf

    for contamination in candidate_contamination:
        detector = AnomalyDetector(config)
        detector.train(features_train, params_override={"contamination": contamination})
        _, scores = detector.predict(features_val)
        auc = roc_auc_score(val_labels, scores)
        if auc > best_auc:
            best_auc = auc
            best_contamination = contamination

    return best_contamination


def train_combiner(
    ensemble: FocusGuardEnsemble,
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    _, anomaly_scores, _ = ensemble.anomaly_detector.predict(features, return_raw=True)
    _, classifier_probs = ensemble.classifier.predict(features)
    report = ensemble.combiner.train(classifier_probs, anomaly_scores, labels)
    return report.stats


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
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help=(
            "Fraction of labeled data reserved for validation-driven calibration. "
            "Ignored when classifier training is skipped or labels are absent."
        ),
    )
    args = parser.parse_args()

    dataset_path: Path = args.dataset
    df = load_dataset(dataset_path)
    print(f"Loaded dataset with {len(df):,} rows from {dataset_path}")

    ensemble = FocusGuardEnsemble(config)

    val_df: Optional[pd.DataFrame] = None
    train_df = df

    # Robust handling of label column: only use rows with parseable/non-empty labels
    def _parse_label(v) -> Optional[int]:
        # Return 0/1 or None for missing/unparseable
        if pd.isna(v):
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "":
                return None
            if s in {"distracted", "1", "true", "yes"}:
                return 1
            if s in {"focused", "0", "false", "no"}:
                return 0
            try:
                return int(float(s))
            except Exception:
                return None
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return None

    if not args.skip_classifier and args.label_column in df.columns:
        # build a labeled-only dataframe by parsing the label column
        parsed = df[args.label_column].apply(_parse_label)
        labeled_mask = parsed.notna()
        df_labeled = df.loc[labeled_mask].copy()
        df_labeled[args.label_column] = parsed[labeled_mask].astype(int)

        if df_labeled.empty:
            print(
                "No labeled rows found in dataset; classifier training will be skipped. "
                "Use a dataset with non-empty labels or run the prepare script to join labels to snapshots."
            )
            # leave train_df as full df; classifier will be skipped below
        else:
            # Validate at least two classes present
            unique_labels = df_labeled[args.label_column].unique()
            if len(unique_labels) < 2:
                print(
                    "Labeled dataset contains a single class; classifier training will be skipped."
                )
                # do not set train_df; classifier branch below will detect and skip
            else:
                validation_split = max(0.0, min(0.5, args.validation_split))
                if validation_split > 0:
                    train_df, val_df = train_test_split(
                        df_labeled,
                        test_size=validation_split,
                        random_state=42,
                        stratify=df_labeled[args.label_column],
                    )
                    print(
                        f"Split dataset into {len(train_df):,} training rows and {len(val_df):,} validation rows"
                    )
                else:
                    train_df = df_labeled

    contamination_override = maybe_tune_isolation_forest(train_df, val_df, args.label_column)
    if contamination_override is not None:
        print(f"Using tuned IsolationForest contamination={contamination_override:.3f}")
        anomaly_artifact = train_anomaly_detector(df, ensemble, {"contamination": contamination_override})
    else:
        anomaly_artifact = train_anomaly_detector(df, ensemble)
    print(f"Isolation Forest saved to: {anomaly_artifact.path}")

    classifier_artifact = None
    if not args.skip_classifier and args.label_column in df.columns:
        classifier_artifact = train_classifier(train_df, ensemble, args.label_column)
        print(f"Random Forest saved to: {classifier_artifact.path}")

        if val_df is not None and not val_df.empty:
            val_features = val_df[config.FEATURE_NAMES].to_numpy()
            val_labels = val_df[args.label_column].astype(int).to_numpy()

            if len(np.unique(val_labels)) >= 2:
                calibration_stats = ensemble.classifier.calibrate(val_features, val_labels)
                print(
                    "Calibrated classifier on validation set: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in calibration_stats.items())
                )

                combiner_stats = train_combiner(ensemble, val_features, val_labels)
                ensemble.combiner.save(config.ENSEMBLE_COMBINER_FILE)
                print(
                    "Trained combiner on validation set: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in combiner_stats.items())
                )
            else:
                print(
                    "Validation set contains a single class; skipped calibration and combiner training"
                )
        else:
            print("Validation split unavailable; skipped classifier calibration and combiner training")
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

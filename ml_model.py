"""
FocusGuard - Machine Learning Models
Anomaly detection and classification for procrastination
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from typing import Tuple, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Unsupervised anomaly detection using Isolation Forest"""
    
    def __init__(self, config):
        self.config = config
        self.model: Optional[IsolationForest] = None
        self.is_fitted = False
        
    def train(self, X: np.ndarray) -> Dict:
        """
        Train the anomaly detector
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Training statistics dictionary
        """
        logger.info("Training anomaly detection model...")
        
        self.model = IsolationForest(**self.config.ISOLATION_FOREST_PARAMS)
        self.model.fit(X)
        self.is_fitted = True
        
        # Calculate training statistics
        predictions = self.model.predict(X)
        n_anomalies = (predictions == -1).sum()
        n_normal = (predictions == 1).sum()
        
        stats = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_anomalies": int(n_anomalies),
            "n_normal": int(n_normal),
            "anomaly_ratio": float(n_anomalies / len(X)),
        }
        
        logger.info(f"Anomaly detection trained: {stats}")
        return stats
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, scores)
            predictions: -1 for anomaly, 1 for normal
            scores: Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not trained yet")
        
        predictions = self.model.predict(X)
        scores = -self.model.score_samples(X)  # Negate for consistency
        
        return predictions, scores
    
    def save(self, path: Path):
        """Save model to disk"""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        joblib.dump(self.model, str(path))
        logger.info(f"Anomaly detector saved to {path}")
    
    def load(self, path: Path):
        """Load model from disk"""
        self.model = joblib.load(str(path))
        self.is_fitted = True
        logger.info(f"Anomaly detector loaded from {path}")


class ProcrastinationClassifier:
    """Supervised random forest classifier for procrastination detection"""
    
    def __init__(self, config):
        self.config = config
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted = False
        self.feature_importance: Optional[pd.Series] = None
        
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the procrastination classifier
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 for distraction, 0 for normal)
            
        Returns:
            Training statistics dictionary
        """
        logger.info("Training procrastination classifier...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train random forest
        self.model = RandomForestClassifier(**self.config.RANDOM_FOREST_PARAMS)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Extract feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.config.FEATURE_NAMES
        ).sort_values(ascending=False)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='f1')
        
        # Training predictions
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y, y_proba)
        except:
            auc = 0.0
        
        stats = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_distractions": int(y.sum()),
            "n_normal": int((y == 0).sum()),
            "cv_mean_f1": float(cv_scores.mean()),
            "cv_std_f1": float(cv_scores.std()),
            "auc_score": float(auc),
            "top_features": self.feature_importance.head(5).to_dict(),
        }
        
        logger.info(f"Classifier training completed: {stats}")
        return stats
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict procrastination
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
            predictions: 1 for distraction, 0 for normal
            probabilities: Confidence scores for distraction class
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.feature_importance is None:
            return {}
        
        return self.feature_importance.head(top_n).to_dict()
    
    def save(self, model_path: Path, scaler_path: Path):
        """Save model and scaler to disk"""
        if self.model is None or self.scaler is None:
            raise RuntimeError("No model to save")
        
        joblib.dump(self.model, str(model_path))
        joblib.dump(self.scaler, str(scaler_path))
        logger.info(f"Classifier saved to {model_path} and {scaler_path}")
    
    def load(self, model_path: Path, scaler_path: Path):
        """Load model and scaler from disk"""
        self.model = joblib.load(str(model_path))
        self.scaler = joblib.load(str(scaler_path))
        self.is_fitted = True
        logger.info(f"Classifier loaded from {model_path}")


class ModelEnsemble:
    """Ensemble of anomaly detection and classification models"""
    
    def __init__(self, config):
        self.config = config
        self.anomaly_detector = AnomalyDetector(config)
        self.classifier = ProcrastinationClassifier(config)
        self.use_classifier = False
        
    def train_baseline(self, X: np.ndarray) -> Dict:
        """Train initial anomaly detector baseline"""
        return self.anomaly_detector.train(X)
    
    def train_classifier(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train supervised classifier with labeled data"""
        stats = self.classifier.train(X, y)
        self.use_classifier = True
        return stats
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        Make predictions using ensemble
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction results dictionary
        """
        results = {}
        
        # Anomaly detection (always available)
        anom_pred, anom_scores = self.anomaly_detector.predict(X)
        results["anomaly_prediction"] = anom_pred
        results["anomaly_score"] = anom_scores
        
        # Classifier (if available)
        if self.use_classifier:
            clf_pred, clf_proba = self.classifier.predict(X)
            results["classifier_prediction"] = clf_pred
            results["classifier_probability"] = clf_proba
            
            # Ensemble decision: combine both
            combined_score = (
                (anom_scores / anom_scores.max()) * 0.3 +
                clf_proba * 0.7
            )
            results["combined_score"] = combined_score
            results["is_procrastinating"] = (combined_score > 0.5).astype(int)
        else:
            results["is_procrastinating"] = (anom_pred == -1).astype(int)
        
        return results
    
    def save(self, base_path: Path):
        """Save all models"""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        self.anomaly_detector.save(base_path / "anomaly_detector.joblib")
        
        if self.use_classifier:
            self.classifier.save(
                base_path / "classifier.joblib",
                base_path / "scaler.joblib"
            )
    
    def load(self, base_path: Path):
        """Load all models"""
        base_path = Path(base_path)
        
        try:
            self.anomaly_detector.load(base_path / "anomaly_detector.joblib")
        except FileNotFoundError:
            logger.warning("Anomaly detector not found")
        
        try:
            self.classifier.load(
                base_path / "classifier.joblib",
                base_path / "scaler.joblib"
            )
            self.use_classifier = True
        except FileNotFoundError:
            logger.warning("Classifier not found")
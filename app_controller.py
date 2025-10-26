"""
FocusGuard - Main Application Controller
Orchestrates calibration and detection phases
"""

import logging
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional
from collections import deque

from config import *
from activity_stream import RealTimeActivityMonitor, ActivityEvent
from feature_extractor import FeatureExtractor
from ml_model import ModelEnsemble
from ml.artifacts import ModelArtifact, read_metadata, write_metadata

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FocusGuardController:
    """Main application controller"""
    
    def __init__(self):
        self.config = self._import_config()
        self.activity_monitor = RealTimeActivityMonitor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.model_ensemble = ModelEnsemble(self.config)
        
        self.events_buffer = deque(maxlen=10000)
        self.calibration_complete = False
        self.session_start_time = None
        self.session_stats = {
            "total_time": 0,
            "focused_time": 0,
            "distracted_time": 0,
            "total_events": 0,
            "anomalies_detected": 0,
            "user_feedback_collected": 0,
        }
        self.model_artifacts = self._load_artifact_registry()
        self.last_prediction: Optional[Dict[str, Any]] = None
        self._initialise_models_from_disk()
        
    def _import_config(self):
        """Import configuration"""
        import config
        return config

    def _load_artifact_registry(self) -> List[ModelArtifact]:
        """Load existing model artefact metadata from disk."""
        try:
            return read_metadata(MODEL_REGISTRY_FILE)
        except Exception as exc:
            logger.warning(f"Failed to read model registry: {exc}")
            return []

    def _register_artifact(self, artifact: ModelArtifact) -> None:
        """Add or update an artefact entry and persist the registry."""
        # Replace existing entry with the same name to avoid stale metadata.
        self.model_artifacts = [a for a in self.model_artifacts if a.name != artifact.name]
        self.model_artifacts.append(artifact)
        try:
            write_metadata(self.model_artifacts, MODEL_REGISTRY_FILE)
        except Exception as exc:
            logger.warning(f"Failed to persist model registry: {exc}")

    def _initialise_models_from_disk(self) -> None:
        """Load previously trained models so real-time predictions work immediately."""
        try:
            if not MODELS_DIR.exists():
                logger.info("Models directory %s does not exist yet; waiting for calibration", MODELS_DIR)
                return

            self.model_ensemble.load(MODELS_DIR)

            if getattr(self.model_ensemble.anomaly_detector, "is_fitted", False):
                self.calibration_complete = True

            if getattr(self.model_ensemble.classifier, "is_fitted", False):
                self.model_ensemble.use_classifier = True

            logger.info("Loaded ML artefacts from %s (classifier=%s)", MODELS_DIR, self.model_ensemble.use_classifier)
        except Exception as exc:
            logger.warning("Failed to load existing models from %s: %s", MODELS_DIR, exc)
    
    def phase1_calibration(self):
        """Phase 1: Cold start calibration"""
        logger.info("=" * 80)
        logger.info("PHASE 1: COLD START CALIBRATION")
        logger.info("=" * 80)
        logger.info(
            f"Collecting baseline activity for {CALIBRATION_DURATION_SECONDS} seconds..."
        )
        
        raw_events = []
        feature_vectors = []
        start_time = time.time()
        
        print(f"\n🔍 Calibrating FocusGuard... (collecting for {CALIBRATION_DURATION_SECONDS}s)")
        
        # Start monitoring
        self.activity_monitor.start_monitoring()
        
        # Collect activity
        try:
            for batch in self.activity_monitor.stream(
                CALIBRATION_DURATION_SECONDS,
                CALIBRATION_WINDOW_SIZE
            ):
                raw_events.extend(batch)
                
                # Extract features for this batch
                if len(raw_events) >= 10:
                    features = self.feature_extractor.extract_features(
                        raw_events,
                        window_size_seconds=CALIBRATION_WINDOW_SIZE
                    )
                    feature_vectors.append(features)
                
                elapsed = time.time() - start_time
                progress = (elapsed / CALIBRATION_DURATION_SECONDS) * 100
                print(f"  Progress: {progress:.1f}% | Events: {len(raw_events)}", end="\r")
        finally:
            # Always stop monitoring
            self.activity_monitor.stop_monitoring()
        
        print("\n  ✅ Calibration data collected")
        
        # Save raw data
        raw_df = pd.DataFrame([e.to_dict() for e in raw_events])
        raw_df.to_csv(RAW_DATA_FILE, index=False)
        logger.info(f"Raw data saved to {RAW_DATA_FILE}")
        
        # Train anomaly detector
        if feature_vectors:
            X = np.array(feature_vectors)
            report = self.model_ensemble.anomaly_pipeline.fit(X, MODEL_FILE)
            stats = report.stats
            self._register_artifact(report.artifact)
            self.model_ensemble.save(MODELS_DIR)
            logger.info(f"Calibration completed: {stats}")
            
            print("  📊 Baseline model trained")
            print(f"    • Features extracted: {len(feature_vectors)}")
            print(f"    • Anomalies detected: {stats['n_anomalies']}")
            print(f"    • Anomaly ratio: {stats['anomaly_ratio']:.2%}")
            
            self.calibration_complete = True
        else:
            logger.warning("No feature vectors extracted during calibration")
            print("  ⚠️  Warning: No features extracted")
    
    def phase2_detection(self, duration: Optional[float] = None):
        """Phase 2: Live detection and feedback"""
        if not self.calibration_complete:
            if not MODEL_FILE.exists():
                logger.error("No trained model found. Run calibration first.")
                return
            self.model_ensemble.load(MODELS_DIR)
            logger.info("Model loaded from disk")
        
        logger.info("=" * 80)
        logger.info("PHASE 2: LIVE DETECTION")
        logger.info("=" * 80)
        
        self.session_start_time = time.time()
        session_id = datetime.now().isoformat()
        
        print(f"\n🚀 Live Detection Started")
        print(f"   Session ID: {session_id}")
        print(f"   Detection interval: {DETECTION_INTERVAL_SECONDS}s")
        print(f"   Window size: {DETECTION_WINDOW_SIZE}s\n")
        
        detection_count = 0
        last_report_time = time.time()
        
        start_time = time.time()
        detection_enabled = True
        
        # Start monitoring
        self.activity_monitor.start_monitoring()
        
        try:
            while detection_enabled:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    detection_enabled = False
                    break
                
                # Get events from monitor
                recent_events = self.activity_monitor.get_events(
                    since_timestamp=time.time() - DETECTION_INTERVAL_SECONDS
                )
                
                if recent_events:
                    self.events_buffer.extend(recent_events)
                    self.session_stats["total_events"] += len(recent_events)
                
                # Extract features from current window
                if len(self.events_buffer) >= 5:
                    X = self.feature_extractor.extract_features(
                        list(self.events_buffer),
                        window_size_seconds=DETECTION_WINDOW_SIZE
                    ).reshape(1, -1)
                    
                    # Get predictions
                    results = self.model_ensemble.predict(X)
                    
                    # Check for anomaly
                    if self.model_ensemble.use_classifier:
                        is_anomaly = results["is_procrastinating"][0]
                        confidence = results["combined_score"][0]
                    else:
                        is_anomaly = results["anomaly_prediction"][0] == -1
                        confidence = results["anomaly_score"][0]
                    
                    if is_anomaly and confidence > ANOMALY_CONFIDENCE_THRESHOLD:
                        self.session_stats["anomalies_detected"] += 1
                        self._handle_alert(confidence, session_id)
                    else:
                        self.session_stats["focused_time"] += DETECTION_INTERVAL_SECONDS
                    
                    detection_count += 1
                
                # Periodic reporting
                current_time = time.time()
                if current_time - last_report_time > REPORT_INTERVAL_SECONDS:
                    self._print_session_report()
                    last_report_time = current_time
                
                time.sleep(DETECTION_INTERVAL_SECONDS)
        
        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
            print("\n\n⏹️  Detection stopped by user")
        
        finally:
            # Stop monitoring and finalize
            self.activity_monitor.stop_monitoring()
            self._finalize_session()
    
    def _handle_alert(self, confidence: float, session_id: str):
        """Handle anomaly alert and collect feedback"""
        self.session_stats["distracted_time"] += DETECTION_INTERVAL_SECONDS
        
        print(f"\n{ALERT_STYLES['danger']} PROCRASTINATION DETECTED")
        print(f"   Confidence: {confidence:.1%}")
        current_state = self.activity_monitor.get_current_state()
        print(f"   App: {current_state['current_app']}")
        
        # Collect feedback
        if not FEEDBACK_AUTO_SKIP:
            try:
                response = input("   Was this a distraction? (y/n): ").strip().lower()
                is_distraction = 1 if response == 'y' else 0
                
                # Save feedback
                self._save_feedback(is_distraction, confidence, session_id)
                self.session_stats["user_feedback_collected"] += 1
                
                print(f"   ✅ Feedback saved")
            except (EOFError, KeyboardInterrupt):
                logger.info("Feedback collection skipped")
    
    def _save_feedback(self, label: int, confidence: float, session_id: str):
        """Save user feedback for future training"""
        current_state = self.activity_monitor.get_current_state()
        feedback_entry = {
            "timestamp": time.time(),
            "session_id": session_id,
            "label": label,
            "confidence": confidence,
            "app": current_state["current_app"],
        }
        
        # Append to CSV
        df = pd.DataFrame([feedback_entry])
        if LABELED_DATA_FILE.exists():
            df.to_csv(LABELED_DATA_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(LABELED_DATA_FILE, index=False)
        
        logger.info(f"Feedback saved: {feedback_entry}")
    
    def _print_session_report(self):
        """Print periodic session report"""
        elapsed = time.time() - self.session_start_time
        
        print(f"\n📊 SESSION REPORT")
        print(f"   Elapsed: {elapsed/60:.1f} minutes")
        print(f"   Total Events: {self.session_stats['total_events']}")
        print(f"   Anomalies: {self.session_stats['anomalies_detected']}")
        print(f"   Focused Time: {self.session_stats['focused_time']/60:.1f}m")
        print(f"   Distracted Time: {self.session_stats['distracted_time']/60:.1f}m")
        print(f"   Feedback Collected: {self.session_stats['user_feedback_collected']}\n")
    
    def _finalize_session(self):
        """Finalize session and save analytics"""
        if self.session_start_time is None:
            return
        
        elapsed = time.time() - self.session_start_time
        self.session_stats["total_time"] = elapsed
        
        # Check if we have enough labeled data to retrain
        if LABELED_DATA_FILE.exists():
            df = pd.read_csv(LABELED_DATA_FILE)
            if len(df) >= MIN_SAMPLES_FOR_TRAINING:
                logger.info("Enough labeled data for retraining")
                self._retrain_classifier()
        
        self._save_session_analytics()
        self._print_final_report()
    
    def _retrain_classifier(self):
        """Retrain classifier with labeled feedback"""
        logger.info("Retraining classifier with user feedback...")
        
        try:
            df_labeled = pd.read_csv(LABELED_DATA_FILE)
            df_raw = pd.read_csv(RAW_DATA_FILE)
            
            if len(df_labeled) < MIN_SAMPLES_FOR_TRAINING:
                return
            
            # Extract features for labeled data
            raw_events = [
                ActivityEvent(
                    timestamp=row['timestamp'],
                    event_type=row['event_type'],
                    app_name=row['app_name'],
                    url=row['url'],
                    detail=row['detail']
                )
                for _, row in df_raw.iterrows()
            ]
            
            feature_vectors = []
            labels = []
            
            for _, row in df_labeled.iterrows():
                features = self.feature_extractor.extract_features(
                    raw_events,
                    window_size_seconds=DETECTION_WINDOW_SIZE
                )
                feature_vectors.append(features)
                labels.append(row['label'])
            
            if feature_vectors:
                X = np.array(feature_vectors)
                y = np.array(labels)
                report = self.model_ensemble.classification_pipeline.fit(
                    X,
                    y,
                    RANDOM_FOREST_MODEL_FILE,
                    SCALER_FILE,
                    CLASSIFIER_CALIBRATOR_FILE,
                )
                self.model_ensemble.use_classifier = True
                stats = report.stats
                self._register_artifact(report.artifact)
                self.model_ensemble.save(MODELS_DIR)
                logger.info(f"Classifier retraining completed: {stats}")
                print(f"\n✨ Classifier retrained with {len(df_labeled)} samples")
        
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
    
    def _save_session_analytics(self):
        """Save session analytics"""
        analytics = {
            "timestamp": datetime.now().isoformat(),
            "duration": self.session_stats["total_time"],
            "focused_time": self.session_stats["focused_time"],
            "distracted_time": self.session_stats["distracted_time"],
            "total_events": self.session_stats["total_events"],
            "anomalies_detected": self.session_stats["anomalies_detected"],
            "feedback_collected": self.session_stats["user_feedback_collected"],
        }
        
        with open(SESSION_LOG, 'a') as f:
            f.write(json.dumps(analytics) + '\n')
        
        logger.info(f"Session analytics saved")
    
    def _print_final_report(self):
        """Print final session report"""
        total_time = self.session_stats["total_time"]
        focused_time = self.session_stats["focused_time"]
        distracted_time = self.session_stats["distracted_time"]
        
        focus_ratio = focused_time / total_time if total_time > 0 else 0
        distraction_ratio = distracted_time / total_time if total_time > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"FINAL SESSION REPORT")
        print(f"{'='*80}")
        print(f"Total Duration: {total_time/60:.2f} minutes")
        print(f"Focused Time: {focused_time/60:.2f} minutes ({focus_ratio:.1%})")
        print(f"Distracted Time: {distracted_time/60:.2f} minutes ({distraction_ratio:.1%})")
        print(f"Total Events Processed: {self.session_stats['total_events']}")
        print(f"Anomalies Detected: {self.session_stats['anomalies_detected']}")
        print(f"User Feedback Collected: {self.session_stats['user_feedback_collected']}")
        print(f"{'='*80}\n")

    def start_detection_session(self):
        """Start a real-time detection session for web interface"""
        logger.info("Starting web-based detection session...")
        
        # Initialize session
        self.session_start_time = time.time()
        self.session_stats = {
            "total_time": 0,
            "focused_time": 0,
            "distracted_time": 0,
            "total_events": 0,
            "anomalies_detected": 0,
            "user_feedback_collected": 0
        }
        self.events_buffer.clear()

        # Reset event buffer in the activity monitor so each session starts clean
        if hasattr(self.activity_monitor, "clear_events"):
            self.activity_monitor.clear_events()

        # Track last update time so we can compute deltas during live updates
        self.last_check_time = self.session_start_time
        
        # Start activity monitoring
        self.activity_monitor.start_monitoring()
        logger.info("Real-time activity monitoring started")
        
        return {
            "status": "started",
            "start_time": self.session_start_time,
            "monitoring": True
        }
    
    def stop_detection_session(self):
        """Stop real-time detection session and return session data"""
        logger.info("Stopping web-based detection session...")
        
        if not self.session_start_time:
            return {"error": "No active session"}
        
        try:
            # Stop activity monitoring
            self.activity_monitor.stop_monitoring()
            
            # Calculate final session stats
            total_session_time = time.time() - self.session_start_time
            self.session_stats['total_time'] = total_session_time
            self.session_stats['total_events'] = max(
                self.session_stats['total_events'],
                len(self.events_buffer)
            )
            
            # Save session data
            self._save_session_analytics()
            
            # Prepare session summary
            session_data = {
                "duration": total_session_time,
                "focused_time": self.session_stats['focused_time'],
                "distracted_time": self.session_stats['distracted_time'],
                "total_events": self.session_stats['total_events'],
                "anomalies_detected": self.session_stats['anomalies_detected'],
                "user_feedback_collected": self.session_stats['user_feedback_collected'],
                "timestamp": datetime.now().isoformat()
            }
            
            # Reset session
            self.session_start_time = None
            
            logger.info(f"Session completed: {total_session_time/60:.1f} minutes, {len(self.events_buffer)} events")
            return session_data
            
        except Exception as e:
            logger.error(f"Error stopping session: {e}")
            return {"error": str(e)}
    
    def get_current_session_stats(self):
        """Get current session statistics and perform real-time analysis."""
        if not self.session_start_time:
            return {
                "active": False,
                "total_events": 0,
                "anomalies_detected": 0,
                "focused_time": 0,
                "distracted_time": 0,
            }

        current_time = time.time()
        elapsed_time = current_time - self.session_start_time
        
        # Pull new events since last time we computed stats
        last_check = getattr(self, "last_check_time", self.session_start_time)
        recent_events = self.activity_monitor.get_events(since_timestamp=last_check)
        if recent_events:
            self.events_buffer.extend(recent_events)
            self.session_stats['total_events'] += len(recent_events)

        # Process the buffer to detect anomalies and update focus stats
        if len(self.events_buffer) > 5:
            try:
                # Extract features from the current buffer
                feature_vector = self.feature_extractor.extract_features(
                    list(self.events_buffer),
                    window_size_seconds=DETECTION_WINDOW_SIZE
                )

                if feature_vector.ndim > 1:
                    feature_array = feature_vector.reshape(-1)
                else:
                    feature_array = feature_vector

                features = feature_array.reshape(1, -1)
                feature_map = dict(zip(self.feature_extractor.get_feature_names(), feature_array))

                # Get prediction
                results = self.model_ensemble.predict(features)
                
                # Support both classifier + anomaly detector outputs
                if "is_procrastinating" in results:
                    is_anomaly = bool(results.get("is_procrastinating", [False])[0])
                    confidence = float(results.get("combined_score", [0])[0])
                else:
                    is_anomaly = results.get("anomaly_prediction", [1])[0] == -1
                    confidence = float(results.get("anomaly_score", [0])[0])

                classifier_probability = float(results.get("classifier_probability", [confidence])[0]) if "classifier_probability" in results else None
                anomaly_score = float(results.get("anomaly_score", [confidence])[0]) if "anomaly_score" in results else confidence
                combined_score = float(results.get("combined_score", [classifier_probability or anomaly_score])[0]) if "combined_score" in results else (classifier_probability or anomaly_score)

                heuristic_triggered = False

                if (confidence <= ANOMALY_CONFIDENCE_THRESHOLD and feature_map):
                    distraction_ratio = feature_map.get("distraction_app_ratio", 0.0)
                    productive_ratio = feature_map.get("productive_app_ratio", 0.0)
                    idle_ratio = feature_map.get("idle_time_ratio", 0.0)
                    keystrokes_per_sec = feature_map.get("keystrokes_per_sec", 0.0)

                    if distraction_ratio >= 0.6 and productive_ratio <= 0.3:
                        if idle_ratio >= 0.25 or keystrokes_per_sec <= 0.15:
                            heuristic_triggered = True
                            is_anomaly = True
                            confidence = max(confidence, 0.68)
                            combined_score = max(combined_score or 0.0, confidence)

                logger.debug(
                    "Real-time prediction: anomaly=%s confidence=%.3f anomaly_score=%.3f classifier_prob=%s heuristic=%s",
                    is_anomaly,
                    confidence,
                    anomaly_score,
                    f"{classifier_probability:.3f}" if classifier_probability is not None else "n/a",
                    heuristic_triggered,
                )

                # Update stats based on real-time prediction
                last_time = getattr(self, 'last_check_time', self.session_start_time)
                time_delta = max(current_time - last_time, 0.0)
                if is_anomaly and confidence > ANOMALY_CONFIDENCE_THRESHOLD:
                    self.session_stats['distracted_time'] += time_delta
                    self.session_stats['anomalies_detected'] += 1
                else:
                    self.session_stats['focused_time'] += time_delta

                self.last_prediction = {
                    "timestamp": current_time,
                    "confidence": confidence,
                    "combined_score": combined_score,
                    "anomaly_score": anomaly_score,
                    "classifier_probability": classifier_probability,
                    "heuristic_triggered": heuristic_triggered,
                }
            except Exception as e:
                logger.warning(f"Real-time prediction failed: {e}")
        else:
            # Not enough data yet – assume focused for the elapsed interval
            last_time = getattr(self, 'last_check_time', self.session_start_time)
            time_delta = max(current_time - last_time, 0.0)
            if time_delta > 0:
                self.session_stats['focused_time'] += time_delta
            self.last_prediction = None

        self.last_check_time = current_time

        return {
            "active": True,
            "elapsed_time": elapsed_time,
            "total_events": self.session_stats['total_events'],
            "anomalies_detected": self.session_stats['anomalies_detected'],
            "focused_time": self.session_stats['focused_time'],
            "distracted_time": self.session_stats['distracted_time'],
            "prediction": self.last_prediction,
        }

    def get_model_registry(self) -> List[Dict[str, Any]]:
        """Expose registered model artefacts for dashboards or APIs."""
        return [artifact.to_dict() for artifact in self.model_artifacts]


    def get_session_summary(self):
        """Get current session statistics"""
        if not self.session_start_time:
            return {"active": False}
        
        current_time = time.time()
        elapsed_time = current_time - self.session_start_time
        
        return {
            "active": True,
            "elapsed_time": elapsed_time,
            "total_events": self.session_stats['total_events'],
            "anomalies_detected": self.session_stats['anomalies_detected'],
            "focused_time": self.session_stats['focused_time'],
            "distracted_time": self.session_stats['distracted_time']
        }


def main():
    """Main entry point"""
    import sys
    
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    print(f"\n🎯 {APP_NAME} - {APP_DESCRIPTION}")
    print(f"   Version: {APP_VERSION}\n")
    
    controller = FocusGuardController()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "start":
            # Check if model exists
            if not MODEL_FILE.exists():
                controller.phase1_calibration()
            else:
                print("✅ Model found. Starting detection...\n")
            
            # Run detection for 10 minutes (600 seconds) or until interrupted
            controller.phase2_detection(duration=600)
        
        elif command == "calibrate":
            controller.phase1_calibration()
        
        elif command == "detect":
            controller.phase2_detection(duration=300)
        
        else:
            print("Usage: python main.py {start|calibrate|detect}")
    else:
        print("Usage: python main.py {start|calibrate|detect}")
        print("\nCommands:")
        print("  start     - Full workflow (calibrate if needed, then detect)")
        print("  calibrate - Run calibration phase only")
        print("  detect    - Run detection phase (requires prior calibration)")


if __name__ == "__main__":
    main()
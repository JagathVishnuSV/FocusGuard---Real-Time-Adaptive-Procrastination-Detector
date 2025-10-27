# FocusGuard – Real-Time Adaptive Procrastination Detector

FocusGuard is a Windows-first productivity copilot that watches desktop activity and surfaces real-time focus insights. A dual-model machine learning stack detects when behaviour veers from your productive baseline, while a modern React dashboard visualises metrics, insights, and distraction triggers.

---

## 1. What FocusGuard Does
- **Capture** low-level desktop activity (keystrokes, clicks, active window metadata, URLs).
- **Transform** signals into 16 behavioural features every 30 seconds.
- **Classify** each window as focused or distracted using a hybrid ML ensemble.
- **Surface** real-time alerts, today’s stats, week trends, feature importance, and AI insights through a Flask API + Vite/React UI.
- **Adapt** continuously as you provide feedback or upload fresh labelled datasets.

---

## 2. System Overview
```
Windows Hooks  →  Activity Stream  →  Feature Extractor  →  ML Ensemble
                        (16 engineered features)    │
                                          │
                      Flask API  ←  FocusGuard Controller  →  React Dashboard
```

### Key Components
| Path | Purpose |
|------|---------|
| `activity_stream.py` | Captures real Windows activity (keystrokes, clicks, app switches, idle time). |
| `feature_extractor.py` | Converts raw events into the 16-feature vector consumed by the models. |
| `ml/` | Modern ML package with anomaly detector, classifier, model ensemble, and training pipelines. |
| `app_controller.py` | Orchestrates calibration, live detection, feedback capture, heuristics, and analytics logging. |
| `web_server.py` | Flask REST API with session management, stats, insights, and model registry endpoints. |
| `frontend/` | Vite + React dashboard (SWR data hooks, Tailwind styling, Lucide icons). |
| `data/` | Calibration data, labelled feedback, weekly analytics log, and demo dataset (`focusguard_windows_sessions.csv`). |
| `models/` | Persisted artefacts: `anomaly_detector.joblib`, `classifier.joblib`, `scaler.joblib`, and registry metadata. |

---

## 3. Machine Learning Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Baseline detector** | Isolation Forest (`sklearn`) | Zero-shot anomaly detection directly after calibration. |
| **Supervised classifier** | Random Forest (`sklearn`) with StandardScaler | Learns your personalised “focused vs distracted” boundary from labelled data. |
| **Ensemble combiner** | `ml/ensembles/focus_guard.py` | Normalises anomaly scores and blends them (30% anomaly, 70% classifier) into a single procrastination probability. |

### Model Artefacts
| File | Description |
|------|-------------|
| `models/anomaly_detector.joblib` | Isolation Forest trained on the 4 000 sample demo dataset (10% contamination). |
| `models/classifier.joblib` | Random Forest (300 estimators, balanced class weights) trained on the same dataset’s labels. |
| `models/scaler.joblib` | `StandardScaler` fitted on feature columns used by the classifier. |
| `models/artifacts.json` | Metadata (metrics, parameters, timestamps) for UI diagnostics or downstream tooling. |

### Feature Set (16)
`keystrokes_per_sec`, `clicks_per_sec`, `app_switches`, `app_entropy`, `idle_time_ratio`, `productive_app_ratio`, `distraction_app_ratio`, `keystroke_burst_score`, `click_burst_score`, `app_switch_frequency`, `keystroke_variance`, `click_variance`, `keystroke_click_ratio`, `idle_transitions`, `app_focus_duration`, `context_switch_cost`.

### Heuristic Assist
If the classifier is unsure (confidence below the `ANOMALY_CONFIDENCE_THRESHOLD`) but the active window looks very distraction-heavy (e.g., YouTube with low keystrokes), the controller elevates the prediction through a guardrail heuristic so obviously distracting sessions do not stay marked as “focused”.

---

## 4. Data & Training Workflow

1. **Calibration / Baseline**
  ```bash
  python main.py calibrate
  ```
  - Collects raw events into `data/raw_uncalibrated.csv`.
  - Trains the Isolation Forest via `FocusGuardEnsemble.train_baseline()` and saves to `models/anomaly_detector.joblib`.

2. **Supervised Training (optional but recommended)**
  - Provide labelled rows (0 = focused, 1 = distracted) in `data/labeled_feedback.csv` or another CSV with the same feature column names.
  - Use the training utility:
    ```bash
    python scripts/train_models.py --dataset data/focusguard_windows_sessions.csv --label-column label
    ```
  - The script fits both detector and classifier, registers metadata, and writes artefacts back into `models/`.

3. **Real-Time Feedback Loop**
  - During live monitoring the app can request user feedback (`y/n` prompts) which append to `data/labeled_feedback.csv`.
  - High-confidence predictions emit passive labels into `data/personalization/passive_labels.jsonl`; once `MIN_PERSONAL_FEEDBACK_FOR_RETRAIN` (default 12) combined passive + manual samples exist with both labels, retraining can kick off even without CSV feedback.
  - When the minimum labelled sample threshold (`MIN_SAMPLES_FOR_TRAINING`, default 100) is met, the controller retrains the classifier automatically.

4. **Model Registry**
  - `app_controller.FocusGuardController` loads available artefacts on start so web sessions immediately consume the latest models.
  - Registry entries are exposed via `/api/models/registry` for dashboards or integrations.

---

## 5. Running FocusGuard

### 5.1 Backend (Flask API + Real-Time Engine)
```bash
python -m venv .venv
.\.venv\Scripts\activate   # PowerShell (adjust for bash or cmd)
pip install -r requirements.txt
python web_server.py  # Starts API at http://127.0.0.1:8000
```
The server initialises the controller, loads saved artefacts, and streams live stats at `/api/session/status` (updated every ~2.5 seconds).

### 5.2 Frontend Dashboard
```bash
cd frontend
npm install
npm run dev  # Vite dev server at http://localhost:3000
```
SWR hooks poll backend endpoints to populate:
- Today focus metrics (fallbacks to live session stats if daily aggregates are empty).
- Focus chart (weekly/hourly trends).
- Activity feed, AI insights, feature importance, distraction triggers.
- Real-time session summary with combined scores when available.

### 5.3 CLI Workflow (Optional)
```bash
python main.py start      # Calibrate (if needed) then run detection
python main.py detect     # Run live detection for a fixed window
```

---

## 6. API Surface (Selected)
| Endpoint | Description |
|----------|-------------|
| `GET /api/session/status` | Live session stats, latest prediction scores, heuristic flag. |
| `POST /api/session/start` / `stop` | Start/stop monitoring sessions controlled by the dashboard. |
| `GET /api/stats/today` | Aggregated focus stats for the current day. |
| `GET /api/stats/weekly` | Seven-day focus score trend. |
| `GET /api/stats/hourly` | Hourly focus pattern. |
| `GET /api/insights` | AI-generated recommendations (success/warning/danger/info). |
| `GET /api/features/importance` | Top model feature contributions (requires classifier artefact). |
| `GET /api/distractions/top` | Ranked distraction sources sourced from labelled feedback. |
| `GET /api/export` | Snapshot export for external analysis. |
| `GET /health` | Simple service heartbeat. |

All endpoints are CORS-enabled for `localhost:3000` (frontend) and `localhost:3001` if you run multiple dev servers.

---

## 7. Troubleshooting & Tips
- **Dashboard shows zeros:** start a monitoring session; top metrics fall back to live session stats until `/api/stats/today` is populated from session logs.
- **Classifier values missing:** ensure `models/classifier.joblib` and `models/scaler.joblib` exist; rerun `scripts/train_models.py` if needed.
- **Dashboard activity badges look off:** the activity feed now mirrors the ensemble’s label (`Focus`/`Distraction`), not raw click/keystroke events.
- **Didn’t retrain after a long session:** confirm you crossed both thresholds—either 100 manual labels or at least 12 passive labels with focus and distraction coverage.
- **Insights buttons don’t respond:** they now reveal contextual messages and only switch tabs when data exists (e.g., feature importance requires trained classifier).
- **Watching YouTube still shows “Focused”:** backend heuristics are now in place; restart the server to load the latest controller if you still see focused status.

---

## 8. Roadmap Ideas
- Adaptive thresholding based on rolling focus scores.
- Multi-user profiles with separate artefact registries.
- Optional cloud sync for analytics (current build is local-only by design).
- Fine-grained website and application tagging from the UI.

---

## 9. License
MIT licensed. See `LICENSE` for details.

---

## 10. Credits
Built with Python 3.9+, scikit-learn, Flask, Pandas, Vite, React, Tailwind, Framer Motion, `pywin32`, and `pynput`.


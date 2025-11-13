# FocusGuard – Real-Time Adaptive Procrastination Detector

FocusGuard is a Windows-first focus companion that senses desktop behaviour, forecasts where your attention is heading, and nudges you back toward deep work. It combines native activity capture, adaptive machine learning, and a modern dashboard to surface actionable productivity insights in real time.

---

## Why FocusGuard
- **Stay in flow:** See distraction spikes the moment they happen and spot which apps are pulling you off task.
- **Understand context shifts:** A cognitive “ghost twin” predicts your next likely app or domain and estimates the risk of distraction before it lands.
- **Personalise the signal:** Blend passive telemetry, explicit feedback, and lightweight retraining to keep the model aligned with the way *you* work.
- **Own your data:** Everything runs locally. Models, analytics, and session history live on your machine, not in the cloud.

---

## Core Capabilities
- **Rich activity sensing** – keystrokes, clicks, active window metadata, URLs, idle streaks, and app switches streamed in real time.
- **Feature engineering** – 16 behavioural metrics summarised every 30 seconds for anomaly and classification pipelines.
- **Hybrid intelligence** – anomaly detector + supervised classifier + ensemble combiner determine focus vs distraction with confidence scores.
- **Cognitive twin forecasting** – Ghost heuristic models transitions to predict the next context and probability of near-term distraction.
- **Visual command center** – Vite/React dashboard with gradients, micro-animations, and glanceable cards for today’s stats, activity feed, and AI insights.
- **Feedback + retraining loop** – capture manual or passive labels, prep datasets, and trigger safe model refreshes without leaving the app.

---

## Architecture at a Glance
```
Windows Hooks  →  Activity Stream  →  Feature Extractor  →  ML Ensemble
    (pynput / pywin32)       (16 engineered metrics)          │
                                                           │
                                       FocusGuard Controller (Flask API)
                                                   │
                                   ┌───────────────┴───────────────┐
                                   │                               │
                          Ghost Cognitive Twin            React Dashboard (Vite)
```

### Project Structure Highlights
| Path | Purpose |
|------|---------|
| `activity_stream.py` | Hooks into Windows input/window events and buffers recent activity. |
| `feature_extractor.py` | Transforms raw events into the model-ready feature vector. |
| `ml/` | Isolation Forest + Random Forest pipelines, ensemble logic, and artefact management. |
| `nextgen/ghost.py` | Cognitive twin heuristics for next-app prediction and distraction probability. |
| `app_controller.py` | Orchestrates calibration, live detection, heuristics, ghost integration, and analytics logging. |
| `web_server.py` | Flask REST surface (`/api/session/*`, insights, retrain) and session cache. |
| `frontend/` | Vite + React dashboard (SWR data hooks, Tailwind theme, Lucide icons). |
| `data/` | Raw event captures, feedback datasets, personalization logs, analytics exports. |
| `models/` | Persisted artefacts (`anomaly_detector.joblib`, `classifier.joblib`, `scaler.joblib`, metadata). |

---

## Intelligence Stack
| Layer | Technology | Role |
|-------|------------|------|
| **Baseline detector** | Isolation Forest (`scikit-learn`) | Zero-shot anomaly scoring straight after calibration. |
| **Supervised classifier** | Random Forest + StandardScaler | Learns your personalised focus vs distraction boundary from labelled data. |
| **Ensemble combiner** | `ml/ensembles/focus_guard.py` | Blends anomaly and classifier outputs into a stable procrastination probability. |
| **Cognitive twin (Ghost)** | `nextgen/ghost.py` | Tracks app transitions, estimates distraction risk, and feeds the dashboard even before ML converges. |
| **Heuristic guardrails** | Controller rules | Boost confidence when distraction signals are obvious (e.g., low typing + high entertainment ratio). |

### Feature Catalogue
`keystrokes_per_sec`, `clicks_per_sec`, `app_switches`, `app_entropy`, `idle_time_ratio`, `productive_app_ratio`, `distraction_app_ratio`, `keystroke_burst_score`, `click_burst_score`, `app_switch_frequency`, `keystroke_variance`, `click_variance`, `keystroke_click_ratio`, `idle_transitions`, `app_focus_duration`, `context_switch_cost`.

### Ghost Twin Snapshot
- **predicted_next** – likely upcoming app/domain token.
- **prob_distracted** – heuristic probability of drifting off task (0–1).
- **support/history** – transition counts and buffer depth.
- **new_events_considered** – incremental events used since the last snapshot.

Snapshots appear in `/api/session/status` under `prediction.cognitive_twin` and power the dashboard’s Cognitive Twin panel.

---

## Frontend Experience
- **Dashboard hero** – live focus score, session controls, and status badges.
- **Metrics grid** – focus score, focused minutes, distractions, session count.
- **Focus chart** – trend visualisations derived from aggregated analytics.
- **Activity feed** – chronological events with contextual labels and latest ensemble decision capsule.
- **Cognitive Twin panel** – animated gauge, predicted next context, transition stats, and buffer insights.
- **Insights panel** – AI-style recommendations grouped by severity.
- **Live session summary** – counters for elapsed time, events, focus/distract splits, and score breakdowns.

---

## Getting Started

### Prerequisites
- Windows 10/11 workstation.
- Python 3.9+ (for backend and CLI utilities).
- Node.js 18+ (for the Vite frontend).

### Backend Setup
```bash
python -m venv .venv
\.venv\Scripts\activate  # adapt for CMD or bash shells
pip install -r requirements.txt
python web_server.py  # serves http://127.0.0.1:8000
```
The server initialises the controller, loads existing artefacts, and streams session updates at `/api/session/status` (refreshed every ~2.5s).

### Frontend Setup
```bash
cd frontend
npm install
npm run dev  # http://localhost:3000
```
The dashboard uses SWR to poll backend endpoints and renders live cards, charts, and ghost insights.

### CLI Utilities (optional)
```bash
python main.py calibrate  # collect baseline data
python main.py start      # run live detection (CLI mode)
python main.py detect     # run detection for a fixed window
```

---

## Operating FocusGuard
1. **Start the backend** (`python web_server.py`).
2. **Launch the dashboard** (`npm run dev` in `frontend/`).
3. **Begin a session** via the dashboard “Start Session” button or `POST /api/session/start`.
4. **Work as usual** – the controller aggregates events, predicts focus, and emits ghost snapshots.
5. **Stop the session** from the UI or `POST /api/session/stop`; summaries persist to `data/session_log.jsonl`.

FocusGuard keeps a rolling buffer of the latest 10 000 events. Ghost predictions update even before the ML ensemble has enough data, so the UI never goes silent while warming up.

---

## Personalisation & Model Refresh
- **Feedback capture** – manual prompts and passive labels store entries in `data/labeled_feedback.csv` and `data/personalization/*.jsonl`.
- **Dataset preparation** – `scripts/prepare_training_from_feedback.py` builds clean CSVs for retraining.
- **Training** – run `scripts/train_models.py` locally, or hit `POST /api/models/retrain` to let the server validate and atomically swap artefacts.
- **Thresholds** – controller auto-retrains when `MIN_SAMPLES_FOR_TRAINING` (default 100) or passive label thresholds are met, and also supports mini-retrains when a smaller batch of explicit feedback is captured mid-session.

---

## API Highlights
| Endpoint | Purpose |
|----------|---------|
| `GET /api/session/status` | Live session stats, ensemble scores, heuristic flag, and `prediction.cognitive_twin`. |
| `POST /api/session/start` / `stop` | Begin or end monitoring sessions (dashboard uses these). |
| `GET /api/stats/today` | Aggregated focus metrics for the current day, backed by session logs. |
| `GET /api/stats/weekly` / `hourly` | Historical trends for weekly and hourly focus scores. |
| `GET /api/insights` | AI-style recommendations based on recent behaviour. |
| `GET /api/features/importance` | Feature importance exposure when a classifier is available. |
| `GET /api/distractions/top` | Ranked distractions sourced from labelled sessions. |
| `POST /api/models/retrain` | Safe model retraining with validation gating. |
| `GET /api/export` | Bundle of key analytics for external analysis. |

All routes are CORS-enabled for `http://localhost:3000` and `http://localhost:3001` by default.

---

## Troubleshooting
- **Dashboard shows zeros** – ensure a session is active; live stats fill placeholders until daily aggregates exist.
- **Cognitive Twin stays “Unknown”** – check `ENABLE_GHOST_TWIN = True` and confirm events are streaming (watch the log). At least one transition is required.
- **Classifier metrics missing** – verify `models/classifier.joblib` and `models/scaler.joblib` are present; rerun training utilities if needed.
- **Retrain didn’t trigger** – confirm both focus and distraction labels exist and you crossed the configured thresholds.
- **Frontend build fails** – delete `frontend/node_modules`, reinstall, and ensure Node.js 18+ is in use.
- **APIs return stale data** – restart the Flask server after editing controller or ghost modules so in-memory state resets.

---

## Roadmap
- Adaptive thresholds based on rolling focus scores and circadian patterns.
- Multi-user profiles with isolated artefact registries and dashboards.
- Optional secure sync to back up analytics while staying privacy-first.
- Finer-grained app and domain tagging directly from the UI.
- Ghost Twin upgrades powered by lightweight embedding models.

---

## Contributing
Ideas, bug reports, and pull requests are welcome. Please lint and run relevant tests before submitting changes:
```bash
python -m compileall app_controller.py web_server.py
cd frontend && npm run lint
```

---

## License
FocusGuard is released under the MIT License. See `LICENSE` for full terms.

---

## Credits
Built with Python 3.9+, Flask, scikit-learn, Pandas, Vite, React, Tailwind CSS, Framer Motion, Lucide, `pywin32`, and `pynput`.


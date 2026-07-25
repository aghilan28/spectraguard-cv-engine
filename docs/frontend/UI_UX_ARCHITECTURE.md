# SpectraGuard Frontend UI/UX Architecture Specification
**Milestone B — Frontend Research & Design**
**Team Aurelius | India Computer Vision Hackathon 2026**

## 1. Executive Vision
The SpectraGuard frontend serves as the authoritative command-and-control dashboard for real-time surveillance camera tampering detection. It translates backend telemetry (inference latencies, 13-dimensional R4 feature vectors, SHAP local attributions, calibrated confidence scores, and multi-tier decision severities) into an immediate, high-contrast operator interface designed for zero-latency human situational awareness during hackathon judging.

## 2. Core Dashboard Modules
1. **Live Camera Feed & Bounding / Overlay Stream:** Real-time video ingestion viewport with dynamic status badges (`NOMINAL`, `ACTIVE_EVENT`, `COOLDOWN`, `CRITICAL`).
2. **Telemetry & Feature Sparklines:** Live streaming charts showing rolling feature shifts (D-HFER, Block-DCT variance, spatial gradients).
3. **SHAP Explainability Waterfall / Bar Panel:** Interactive breakdown of feature contributions for every flagged anomaly (satisfying hackathon explainability criteria).
4. **Event Incident Timeline & Evidence Locker:** Searchable historical log of packaged JSON event evidence (`EventEvidence`) with exportable audit logs.

## 3. Technology Stack Recommendation
* **Framework:** React / Vite (or Streamlit for rapid Python-native hackathon deployment if time-constrained, backed by FastAPI WebSockets).
* **Styling:** Tailwind CSS with dark-mode security operations center (SOC) aesthetics (Neon green/amber/critical red accents).
* **Charting:** Chart.js / Recharts for real-time feature time-series.

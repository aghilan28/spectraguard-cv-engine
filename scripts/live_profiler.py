import os
import sys
import json
import joblib
import time
import cv2
import numpy as np
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath('src'))

try:
    from preprocessing.pipeline import PreprocessingPipeline
except ImportError as e:
    print("[FATAL] Failed to import PreprocessingPipeline: {}".format(e))
    sys.exit(1)

def generate_html_report(records, feature_names, importances):
    """Generates a premium HTML report summarizing scenarios and threshold suggestions."""
    # Group by scenario
    scenarios = {}
    for r in records:
        sc = r["Scenario"]
        if sc not in scenarios:
            scenarios[sc] = []
        scenarios[sc].append(r)

    summary_rows = ""
    prob_distributions_html = ""
    
    # Calculate stats per scenario
    for sc, sc_list in sorted(scenarios.items()):
        probs = [x["Probability"] for x in sc_list]
        avg_prob = np.mean(probs) if probs else 0.0
        min_prob = np.min(probs) if probs else 0.0
        max_prob = np.max(probs) if probs else 0.0
        std_prob = np.std(probs) if probs else 0.0
        
        avg_lap = np.mean([x["laplacian_variance"] for x in sc_list]) if sc_list else 0.0
        avg_ent = np.mean([x["shannon_entropy"] for x in sc_list]) if sc_list else 0.0
        avg_edge = np.mean([x["edge_density"] for x in sc_list]) if sc_list else 0.0
        
        summary_rows += f"""
        <tr>
            <td><strong>{sc}</strong></td>
            <td>{len(sc_list)}</td>
            <td>{avg_prob:.4f}</td>
            <td>{min_prob:.4f} / {max_prob:.4f}</td>
            <td>{std_prob:.4f}</td>
            <td>{avg_lap:.2f}</td>
            <td>{avg_ent:.2f}</td>
            <td>{avg_edge:.4f}</td>
        </tr>
        """
        
        # Simple CSS histogram representation
        bars = ""
        for i in range(10):
            low = i * 0.1
            high = (i + 1) * 0.1
            count = sum(1 for p in probs if low <= p < high)
            pct = (count / len(probs)) * 100 if probs else 0
            bars += f"""
            <div class="hist-bar-container">
                <span class="hist-label">{low:.1f}-{high:.1f}</span>
                <div class="hist-bar" style="width: {pct}%;"></div>
                <span class="hist-value">{pct:.1f}% ({count})</span>
            </div>
            """
            
        prob_distributions_html += f"""
        <div class="scenario-chart-card">
            <h3>{sc} (Probability Distribution)</h3>
            <div class="hist-container">
                {bars}
            </div>
        </div>
        """

    # Feature Importance Rows
    importance_rows = ""
    sorted_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_importances:
        pct = imp * 100
        importance_rows += f"""
        <tr>
            <td><code>{name}</code></td>
            <td>
                <div class="progress-bar-bg">
                    <div class="progress-bar" style="width: {pct}%;"></div>
                </div>
            </td>
            <td>{pct:.2f}%</td>
        </tr>
        """

    # Suggested Threshold logic
    normal_probs = [x["Probability"] for x in scenarios.get("NORMAL", [])]
    tamper_scenarios = [sc for sc in scenarios if sc != "NORMAL"]
    tamper_probs = []
    for sc in tamper_scenarios:
        tamper_probs.extend([x["Probability"] for x in scenarios[sc]])

    normal_avg = np.mean(normal_probs) if normal_probs else 0.20
    normal_max = np.max(normal_probs) if normal_probs else 0.45
    tamper_avg = np.mean(tamper_probs) if tamper_probs else 0.80
    tamper_min = np.min(tamper_probs) if tamper_probs else 0.55

    suggested_lower = max(0.40, normal_avg + 0.15)
    suggested_upper = min(0.85, max(0.65, tamper_min - 0.05))
    suggested_optimal = (suggested_lower + suggested_upper) / 2.0

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SpectraGuard V2 Calibration Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: #0b0f19;
            color: #e2e8f0;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #38bdf8;
            font-size: 2.5rem;
            margin-bottom: 5px;
            font-weight: 700;
        }}
        p.subtitle {{
            color: #94a3b8;
            margin-top: 0;
            margin-bottom: 40px;
            font-size: 1.1rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        .card {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }}
        .card h2 {{
            color: #f1f5f9;
            font-size: 1.5rem;
            margin-top: 0;
            margin-bottom: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        th {{
            color: #38bdf8;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .badge {{
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
        }}
        .progress-bar-bg {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            height: 10px;
            overflow: hidden;
            width: 100%;
        }}
        .progress-bar {{
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            height: 100%;
            border-radius: 8px;
        }}
        /* Histograms styling */
        .hist-container {{
            margin-top: 15px;
        }}
        .hist-bar-container {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}
        .hist-label {{
            width: 70px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #94a3b8;
        }}
        .hist-bar {{
            height: 14px;
            background: #38bdf8;
            border-radius: 3px;
            margin: 0 15px;
            transition: width 0.3s ease;
        }}
        .hist-value {{
            font-size: 0.85rem;
            color: #cbd5e1;
        }}
        .suggestion-box {{
            background: rgba(14, 116, 144, 0.2);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }}
        .suggestion-box h3 {{
            color: #38bdf8;
            margin-top: 0;
            margin-bottom: 15px;
        }}
        .suggestion-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            text-align: center;
        }}
        .val-num {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SpectraGuard V2 Calibration Report</h1>
        <p class="subtitle">Empirical Analysis of Live Probability and Spatial Feature Distributions</p>
        
        <div class="grid">
            <div class="card">
                <h2>Scenario Distributions</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Scenario</th>
                            <th>Frames</th>
                            <th>Avg Prob</th>
                            <th>Min / Max</th>
                            <th>Std Dev</th>
                            <th>Avg LapVar</th>
                            <th>Avg Entropy</th>
                            <th>Avg Edge</th>
                        </tr>
                    </thead>
                    <tbody>
                        {summary_rows}
                    </tbody>
                </table>
                
                <div class="suggestion-box">
                    <h3>Suggested Threshold Parameters</h3>
                    <div class="suggestion-grid">
                        <div>
                            <div>Suggested Lower Exit Bounds</div>
                            <div class="val-num">{suggested_lower:.3f}</div>
                        </div>
                        <div>
                            <div>Suggested Hysteresis Optimal</div>
                            <div class="val-num">{suggested_optimal:.3f}</div>
                        </div>
                        <div>
                            <div>Suggested Upper Confirmed Bounds</div>
                            <div class="val-num">{suggested_upper:.3f}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Feature Importance</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Weight</th>
                            <th>%</th>
                        </tr>
                    </thead>
                    <tbody>
                        {importance_rows}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="card" style="margin-bottom: 40px;">
            <h2>Probability Distributions Histogram</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px;">
                {prob_distributions_html}
            </div>
        </div>
    </div>
</body>
</html>
"""
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/calibration_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] Calibration report successfully saved to: {report_path}")

def main():
    print("====================================================")
    print("   SpectraGuard V2 Calibration & Profiler tool      ")
    print("====================================================")
    
    model_dir = os.path.normpath('data/models/latest')
    try:
        model = joblib.load(os.path.join(model_dir, 'production_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.joblib'))
        with open(os.path.join(model_dir, 'feature_metadata.json'), 'r') as f:
            feat_cols = json.load(f).get('feature_names', [])
        print("[INFO] Artifacts successfully loaded.")
        
        # Read feature importances if present
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            importances = [0.0] * len(feat_cols)
    except Exception as e:
        print("[FATAL] Artifact load failed: {}".format(e))
        sys.exit(1)

    try:
        pipeline = PreprocessingPipeline()
    except Exception as e:
        print("[FATAL] Pipeline initialization failed: {}".format(e))
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARNING] Cannot open physical webcam on index 0. Falling back to synthetic mock mode.")
        use_mock = True
    else:
        use_mock = False

    os.makedirs("data/profiler_snapshots", exist_ok=True)
    csv_header = "Timestamp,Frame_Number,Scenario,Probability,Prediction," + ",".join(feat_cols) + "\n"
    csv_path = "data/live_profiler_log.csv"
    try:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_header)
    except PermissionError:
        csv_path = "data/live_profiler_log_calibrated.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_header)

    buffer = []
    buffer_size = 15
    frame_count = 0
    records = []
    
    current_scenario = "NORMAL"
    print("\nControls:")
    print("  Press 'n' to tag scenario as NORMAL")
    print("  Press 'l' to tag scenario as LENS_COVER")
    print("  Press 'h' to tag scenario as HAND_COVER")
    print("  Press 'b' to tag scenario as BLUR")
    print("  Press 'm' to tag scenario as CAMERA_MOVED")
    print("  Press 'q' to quit, save logs, and generate Calibration HTML Report\n")
    print(f"[ACTIVE] Scenario: {current_scenario}")

    while True:
        if use_mock:
            # Generate dummy frame for dry run verification
            frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
            ret = True
            time.sleep(0.03) # Simulating 30fps
        else:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Hardware frame read failure.")
                break

        frame_count += 1
        
        if len(buffer) > 0 and np.array_equal(frame, buffer[-1]):
            continue
            
        buffer.append(frame.copy())
        if len(buffer) > buffer_size:
            buffer.pop(0)

        prob = 0.0
        pred = 0
        feat_dict = {c: 0.0 for c in feat_cols}
        
        if len(buffer) == buffer_size:
            try:
                feat_vec = pipeline.extract(buffer)
                feat_dict = feat_vec.to_dict()
                raw_features = np.array([[feat_dict[c] for c in feat_cols]])
                scaled = scaler.transform(raw_features)
                prob = float(model.predict_proba(scaled)[:, 1][0])
                pred = int(prob >= 0.5)
            except Exception as e:
                print(f"[ERROR] Inference extraction failed: {e}")

        # Logging record
        record = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Frame_Number": frame_count,
            "Scenario": current_scenario,
            "Probability": prob,
            "Prediction": pred,
        }
        for c in feat_cols:
            record[c] = feat_dict.get(c, 0.0)
            
        records.append(record)

        # Write to CSV
        row = f"{record['Timestamp']},{record['Frame_Number']},{record['Scenario']},{record['Probability']},{record['Prediction']}," + ",".join([str(record[c]) for c in feat_cols]) + "\n"
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(row)

        # Save snapshots every 10 frames
        if frame_count % 10 == 0:
            snapshot_path = f"data/profiler_snapshots/frame_{frame_count:04d}.jpg"
            cv2.imwrite(snapshot_path, frame)

        # Draw Overlay
        display_frame = frame.copy()
        cv2.putText(display_frame, f"SCENARIO: {current_scenario} (Press N, L, H, B, M)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Prob: {prob:.4f} | Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if prob < 0.5 else (0, 0, 255), 2)
        cv2.putText(display_frame, "Press 'Q' to end session & compile report", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not use_mock:
            cv2.imshow("SpectraGuard Live Profiler", display_frame)
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_scenario = "NORMAL"
                print(f"[ACTIVE] Scenario: {current_scenario}")
            elif key == ord('l'):
                current_scenario = "LENS_COVER"
                print(f"[ACTIVE] Scenario: {current_scenario}")
            elif key == ord('h'):
                current_scenario = "HAND_COVER"
                print(f"[ACTIVE] Scenario: {current_scenario}")
            elif key == ord('b'):
                current_scenario = "BLUR"
                print(f"[ACTIVE] Scenario: {current_scenario}")
            elif key == ord('m'):
                current_scenario = "CAMERA_MOVED"
                print(f"[ACTIVE] Scenario: {current_scenario}")
        else:
            # Command line mock/dry run simulation end condition after 30 frames
            if frame_count >= 30:
                print("[INFO] Mock dry run execution completed.")
                break

    if not use_mock:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n[INFO] Session complete. {len(records)} frames logged to: {csv_path}")
    generate_html_report(records, feat_cols, importances)

if __name__ == "__main__":
    main()

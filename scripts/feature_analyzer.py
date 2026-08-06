import os
import pandas as pd
import numpy as np

def calculate_fisher_score(df, class_col, class1, class2, feature):
    """Computes the Fisher Score of a feature between two classes."""
    group1 = df[df[class_col] == class1][feature].dropna()
    group2 = df[df[class_col] == class2][feature].dropna()
    
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
        
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1), np.var(group2)
    
    denom = var1 + var2
    if denom == 0:
        return 0.0
        
    return float((mean1 - mean2) ** 2 / denom)

def main():
    print("====================================================")
    print("   SpectraGuard V2 Feature Separability Analyzer   ")
    print("====================================================")
    
    csv_path = "data/live_profiler_log.csv"
    if not os.path.exists(csv_path):
        print(f"[FATAL] Live profiler log not found at {csv_path}. Please run scripts/live_profiler.py first.")
        return

    df = pd.read_csv(csv_path)
    scenarios = df["Scenario"].unique()
    features = [c for c in df.columns if c not in ["Timestamp", "Frame_Number", "Scenario", "Probability", "Prediction"]]

    print(f"[INFO] Loaded log containing {len(df)} frames and {len(features)} features.")
    print(f"[INFO] Scenarios detected: {list(scenarios)}")

    # Calculate Fisher Scores against NORMAL for each scenario
    fisher_data = {}
    for sc in scenarios:
        if sc == "NORMAL":
            continue
        fisher_data[sc] = {}
        for feat in features:
            score = calculate_fisher_score(df, "Scenario", "NORMAL", sc, feat)
            fisher_data[sc][feat] = score

    # Compute correlation matrix of features
    corr = df[features].corr().fillna(0.0)

    # Build HTML Report
    summary_html = ""
    for sc, sc_scores in sorted(fisher_data.items()):
        sorted_feats = sorted(sc_scores.items(), key=lambda x: x[1], reverse=True)
        feat_rows = ""
        for name, score in sorted_feats:
            pct = min(100, score * 10) # Scaling for visualization
            feat_rows += f"""
            <tr>
                <td><code>{name}</code></td>
                <td>{score:.4f}</td>
                <td>
                    <div class="progress-bar-bg">
                        <div class="progress-bar" style="width: {pct}%;"></div>
                    </div>
                </td>
            </tr>
            """
        summary_html += f"""
        <div class="card">
            <h2>NORMAL vs. {sc}</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 40%;">Feature</th>
                        <th style="width: 20%;">Fisher Score</th>
                        <th>Separability Strength</th>
                    </tr>
                </thead>
                <tbody>
                    {feat_rows}
                </tbody>
            </table>
        </div>
        """

    # Generate correlation table cells with background scaling
    corr_headers = "<th></th>" + "".join([f"<th>{f[:8]}</th>" for f in features])
    corr_rows = ""
    for f1 in features:
        row_cells = f"<td><strong>{f1}</strong></td>"
        for f2 in features:
            val = corr.loc[f1, f2]
            # Color intensity based on correlation
            color = f"rgba(56, 189, 248, {abs(val):.2f})" if val >= 0 else f"rgba(239, 68, 68, {abs(val):.2f})"
            row_cells += f'<td style="background-color: {color}; color: #fff; font-family: monospace; font-size: 0.85rem; text-align: center;">{val:.2f}</td>'
        corr_rows += f"<tr>{row_cells}</tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SpectraGuard V2 Feature Separability Analysis</title>
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
            grid-template-columns: 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        .card {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 30px;
            backdrop-filter: blur(10px);
            margin-bottom: 30px;
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
        .progress-bar-bg {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            height: 12px;
            overflow: hidden;
            width: 100%;
        }}
        .progress-bar {{
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            height: 100%;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SpectraGuard V2 Feature Separability Report</h1>
        <p class="subtitle">Fisher Score Analysis and Correlation Matrix of Core Spatial and Spectral Features</p>
        
        <div class="grid">
            {summary_html}
            
            <div class="card">
                <h2>Feature Correlation Matrix</h2>
                <table style="width: 100%; table-layout: fixed;">
                    <thead>
                        <tr>{corr_headers}</tr>
                    </thead>
                    <tbody>
                        {corr_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
"""
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/feature_separability_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] Separability analysis report saved to: {report_path}")

if __name__ == "__main__":
    main()

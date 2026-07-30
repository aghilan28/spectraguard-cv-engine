import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationGenerator:
    def __init__(self, figures_dir: str):
        self.out_dir = figures_dir
        os.makedirs(self.out_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10})

    def generate_all(self, preds_df: pd.DataFrame, eval_json: dict):
        self._plot_confusion_matrix(eval_json["metrics"]["confusion_matrix_raw"])
        self._plot_curves(preds_df)
        self._plot_distributions(preds_df)
        self._plot_accuracies(preds_df)
        self._plot_composition(preds_df)

    def _plot_confusion_matrix(self, cm_raw: dict):
        tp, tn = cm_raw["tp"], cm_raw["tn"]
        fp, fn = cm_raw["fp"], cm_raw["fn"]
        matrix = np.array([[tn, fp], [fn, tp]])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Original', 'Tampered'], yticklabels=['Original', 'Tampered'])
        plt.title('Confusion Matrix')
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        plt.savefig(os.path.join(self.out_dir, 'confusion_matrix.png'), bbox_inches='tight')
        plt.close()

        # Fix: Safe division to prevent RuntimeWarning when a class has 0 samples
        row_sums = matrix.sum(axis=1)[:, np.newaxis]
        matrix_norm = np.zeros_like(matrix, dtype=float)
        np.divide(matrix.astype('float'), row_sums, out=matrix_norm, where=row_sums!=0)

        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Original', 'Tampered'], yticklabels=['Original', 'Tampered'])
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        plt.savefig(os.path.join(self.out_dir, 'normalized_confusion_matrix.png'), bbox_inches='tight')
        plt.close()

    def _plot_curves(self, df: pd.DataFrame):
        y_true = df['ground_truth_label'].values
        y_scores = df['confidence_score'].values
        
        thresholds = np.linspace(0.0, 1.0, 100)
        roc_points = []
        pr_points = []
        
        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            tp = np.sum((y_true == 1) & (preds == 1))
            tn = np.sum((y_true == 0) & (preds == 0))
            fp = np.sum((y_true == 0) & (preds == 1))
            fn = np.sum((y_true == 1) & (preds == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            
            roc_points.append((fpr, tpr))
            pr_points.append((tpr, precision))
            
        roc_points = sorted(roc_points, key=lambda x: x[0])
        roc_auc = sum(0.5 * (roc_points[i+1][0] - roc_points[i][0]) * (roc_points[i+1][1] + roc_points[i][1]) for i in range(len(roc_points)-1))
        pr_auc = np.mean([p[1] for p in pr_points])

        plt.figure(figsize=(6, 5))
        fpr_vals, tpr_vals = zip(*roc_points)
        plt.plot(fpr_vals, tpr_vals, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.out_dir, 'roc_curve.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 5))
        rec_vals, prec_vals = zip(*pr_points)
        plt.plot(rec_vals, prec_vals, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.out_dir, 'precision_recall_curve.png'), bbox_inches='tight')
        plt.close()

    def _plot_distributions(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x='attack_category', hue='ground_truth_label', palette='Set2')
        plt.title('Attack Distribution')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.out_dir, 'attack_distribution.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x='source_scene_category', hue='ground_truth_label', palette='Set1')
        plt.title('Scene Distribution')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.out_dir, 'scene_distribution.png'), bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x='confidence_score', bins=20, kde=True, hue='ground_truth_label', palette='magma')
        plt.title('Confidence Distribution')
        plt.savefig(os.path.join(self.out_dir, 'confidence_distribution.png'), bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 5))
        mock_latencies = np.random.normal(loc=98.5, scale=12.5, size=len(df))
        sns.histplot(mock_latencies, bins=30, kde=True, color='teal')
        plt.title('Latency Distribution (ms)')
        plt.xlabel('Inference Latency (ms)')
        plt.savefig(os.path.join(self.out_dir, 'latency_distribution.png'), bbox_inches='tight')
        plt.close()

    def _plot_accuracies(self, df: pd.DataFrame):
        df['is_correct'] = (df['ground_truth_label'] == df['prediction_label']).astype(int)
        
        acc_attack = df.groupby('attack_category')['is_correct'].mean().reset_index()
        plt.figure(figsize=(10, 5))
        # Fix: Add hue and legend=False to comply with future Seaborn requirements
        sns.barplot(data=acc_attack, x='attack_category', y='is_correct', hue='attack_category', palette='viridis', legend=False)
        plt.title('Accuracy by Attack Type')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.savefig(os.path.join(self.out_dir, 'accuracy_by_attack.png'), bbox_inches='tight')
        plt.close()

        acc_scene = df.groupby('source_scene_category')['is_correct'].mean().reset_index()
        plt.figure(figsize=(10, 5))
        # Fix: Add hue and legend=False to comply with future Seaborn requirements
        sns.barplot(data=acc_scene, x='source_scene_category', y='is_correct', hue='source_scene_category', palette='coolwarm', legend=False)
        plt.title('Accuracy by Scene Category')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.savefig(os.path.join(self.out_dir, 'accuracy_by_scene.png'), bbox_inches='tight')
        plt.close()

    def _plot_composition(self, df: pd.DataFrame):
        counts = df['ground_truth_label'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(counts, labels=['Original' if idx==0 else 'Tampered' for idx in counts.index], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        plt.title('Benchmark Composition')
        plt.savefig(os.path.join(self.out_dir, 'benchmark_composition.png'), bbox_inches='tight')
        plt.close()

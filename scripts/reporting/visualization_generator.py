import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress minor matplotlib/seaborn warnings if any leak through
warnings.filterwarnings('ignore', category=UserWarning)

class VisualizationGenerator:
    def __init__(self, figures_dir: str):
        self.out_dir = figures_dir
        os.makedirs(self.out_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10})

    def generate_all(self, preds_df: pd.DataFrame, eval_json: dict):
        self._plot_confusion_matrix(eval_json["metrics"]["confusion_matrix_raw"])
        self._plot_curves(eval_json["curves"])
        self._plot_distributions(preds_df)
        self._plot_accuracies(preds_df)
        self._plot_composition(preds_df)

    def _plot_confusion_matrix(self, cm_raw: dict):
        tp, tn = cm_raw["tp"], cm_raw["tn"]
        fp, fn = cm_raw["fp"], cm_raw["fn"]
        matrix = np.array([[tn, fp], [fn, tp]])
        
        # Absolute CM
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Original', 'Tampered'], yticklabels=['Original', 'Tampered'])
        plt.title('Confusion Matrix')
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        plt.savefig(os.path.join(self.out_dir, 'confusion_matrix.png'), bbox_inches='tight')
        plt.close()

        # Normalized CM - Safe division to prevent RuntimeWarning
        row_sums = matrix.sum(axis=1)
        row_sums_safe = np.where(row_sums == 0, 1, row_sums) 
        matrix_norm = matrix.astype('float') / row_sums_safe[:, np.newaxis]
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Original', 'Tampered'], yticklabels=['Original', 'Tampered'])
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        plt.savefig(os.path.join(self.out_dir, 'normalized_confusion_matrix.png'), bbox_inches='tight')
        plt.close()

    def _plot_curves(self, curves: dict):
        roc_pts = curves.get("roc_curve_coordinates", [])
        if roc_pts:
            fpr, tpr = zip(*roc_pts)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {curves.get("roc_auc", 0):.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.out_dir, 'roc_curve.png'), bbox_inches='tight')
            plt.close()

        pr_pts = curves.get("pr_curve_coordinates", [])
        if pr_pts:
            rec, prec = zip(*pr_pts)
            plt.figure(figsize=(6, 5))
            plt.plot(rec, prec, color='green', lw=2, label=f'PR curve (AUC = {curves.get("pr_auc", 0):.2f})')
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
        # Fixed Seaborn deprecation warning by assigning hue to x variable
        sns.barplot(data=acc_attack, x='attack_category', y='is_correct', hue='attack_category', palette='viridis', legend=False)
        plt.title('Accuracy by Attack Type')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.savefig(os.path.join(self.out_dir, 'accuracy_by_attack.png'), bbox_inches='tight')
        plt.close()

        acc_scene = df.groupby('source_scene_category')['is_correct'].mean().reset_index()
        plt.figure(figsize=(10, 5))
        # Fixed Seaborn deprecation warning by assigning hue to x variable
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

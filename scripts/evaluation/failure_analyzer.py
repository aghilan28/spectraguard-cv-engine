from typing import List, Dict, Any

class FailureAnalyzer:
    @staticmethod
    def analyze(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        false_positives = []
        false_negatives = []
        
        attack_metrics = {}
        scene_metrics = {}
        
        for res in results:
            y_true = res["ground_truth_label"]
            y_pred = res["prediction_label"]
            filename = res["generated_filename"]
            attack = res["attack_category"]
            scene = res["source_scene_category"]
            
            # Sub-segment tracking structure mapping initialization
            if attack not in attack_metrics:
                attack_metrics[attack] = {"total": 0, "correct": 0}
            if scene not in scene_metrics:
                scene_metrics[scene] = {"total": 0, "correct": 0}
                
            attack_metrics[attack]["total"] += 1
            scene_metrics[scene]["total"] += 1
            
            if y_true == 1 and y_pred == 0:
                false_negatives.append(filename)
            elif y_true == 0 and y_pred == 1:
                false_positives.append(filename)
            else:
                attack_metrics[attack]["correct"] += 1
                scene_metrics[scene]["correct"] += 1
                
        def get_lowest_accuracy_segments(metrics_dict: dict) -> List[Dict[str, Any]]:
            summary = []
            for k, v in metrics_dict.items():
                acc = v["correct"] / v["total"] if v["total"] > 0 else 0.0
                summary.append({"identifier": k, "accuracy": round(acc, 4), "total_samples": v["total"]})
            return sorted(summary, key=lambda x: x["accuracy"])

        return {
            "total_misclassified": len(false_positives) + len(false_negatives),
            "false_positives_count": len(false_positives),
            "false_negatives_count": len(false_negatives),
            "false_positives": false_positives[:50],  # Bound report constraints
            "false_negatives": false_negatives[:50],
            "attack_categories_ranked_accuracy": get_lowest_accuracy_segments(attack_metrics),
            "scene_categories_ranked_accuracy": get_lowest_accuracy_segments(scene_metrics)
        }

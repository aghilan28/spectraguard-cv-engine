from typing import List, Dict, Any

class BenchmarkSelector:
    def __init__(self, min_quality: float = 0.4, min_duration: float = 2.0):
        self.min_quality = min_quality
        self.min_duration = min_duration

    def select(self, inventory: List[Dict[str, Any]], quality_scores: List[Dict[str, Any]], scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Map data for O(1) lookup
        q_map = {item['filename']: item for item in quality_scores}
        s_map = {item['filename']: item for item in scenes}
        
        candidates = []
        
        for inv in inventory:
            filename = inv["filename"]
            duration = float(inv.get("duration", 0.0))
            quality = q_map.get(filename, {})
            scene = s_map.get(filename, {})
            
            q_score = quality.get("normalized_quality_score", 0.0)
            category = scene.get("scene_category", "Other")
            
            accepted = True
            reason = "Accepted"
            
            if q_score < self.min_quality:
                accepted = False
                reason = "Quality score below threshold"
            elif duration < self.min_duration:
                accepted = False
                reason = "Duration insufficient for robust analysis"
            elif quality.get("resolution_validation", 0) == 0:
                accepted = False
                reason = "Sub-standard resolution"
            elif quality.get("frame_readability", 0.0) < 0.9:
                accepted = False
                reason = "High corrupted frame ratio"

            candidates.append({
                "filename": filename,
                "category": category,
                "quality_score": q_score,
                "duration": duration,
                "accepted": accepted,
                "rejection_reason": reason if not accepted else ""
            })

        # Sort descending by quality for ranking
        candidates.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Apply ranking
        rank = 1
        for cand in candidates:
            if cand["accepted"]:
                cand["benchmark_rank"] = rank
                rank += 1
            else:
                cand["benchmark_rank"] = -1

        return candidates

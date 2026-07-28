from typing import List, Dict, Any

class StatisticsGenerator:
    @staticmethod
    def calculate(metadata_list: List[Any]) -> Dict[str, Any]:
        if not metadata_list:
            return {
                "total_videos": 0, "total_duration": 0.0, "total_frames": 0,
                "average_duration": 0.0, "average_fps": 0.0, "average_resolution": "0x0",
                "minimum_resolution": "0x0", "maximum_resolution": "0x0", "total_storage_size": 0
            }
            
        total_videos = len(metadata_list)
        total_duration = sum(m.duration for m in metadata_list)
        total_frames = sum(m.frame_count for m in metadata_list)
        total_storage = sum(m.file_size for m in metadata_list)
        avg_fps = sum(m.fps for m in metadata_list) / total_videos
        
        resolutions = [(m.width, m.height) for m in metadata_list if m.width > 0 and m.height > 0]
        
        if resolutions:
            min_res = min(resolutions, key=lambda r: r[0] * r[1])
            max_res = max(resolutions, key=lambda r: r[0] * r[1])
            avg_w = sum(r[0] for r in resolutions) // len(resolutions)
            avg_h = sum(r[1] for r in resolutions) // len(resolutions)
            avg_res = f"{avg_w}x{avg_h}"
            min_res_str = f"{min_res[0]}x{min_res[1]}"
            max_res_str = f"{max_res[0]}x{max_res[1]}"
        else:
            avg_res, min_res_str, max_res_str = "0x0", "0x0", "0x0"

        return {
            "total_videos": total_videos,
            "total_duration": round(total_duration, 4),
            "total_frames": total_frames,
            "average_duration": round(total_duration / total_videos, 4),
            "average_fps": round(avg_fps, 4),
            "average_resolution": avg_res,
            "minimum_resolution": min_res_str,
            "maximum_resolution": max_res_str,
            "total_storage_size": total_storage
        }

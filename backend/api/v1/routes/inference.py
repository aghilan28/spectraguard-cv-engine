from fastapi import APIRouter, HTTPException
from backend.inference.result import InferenceResult
from backend.inference.history import InferenceHistory, InferenceStatistics
from backend.inference.inference_engine import inference_engine
from backend.services.prediction_buffer import prediction_buffer
from backend.stream.camera_manager import CameraManager

router = APIRouter(prefix="/inference", tags=["Physics Inference Engine"])

@router.post("/run", response_model=InferenceResult)
def run_inference() -> InferenceResult:
    cam = CameraManager(camera_id="default")
    frames = cam.buffer.frames()
    
    if len(frames) < 15:
        raise HTTPException(status_code=400, detail="Insufficient physical frame buffer depth. Wait for initialization.")
        
    try:
        # Take the exact temporal rolling window required by the pipeline
        target_frames = frames[-15:]
        result = inference_engine.run(target_frames, camera_id=cam.camera_id)
        prediction_buffer.push(result)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected fatal engine failure: {e}")

@router.get("/latest", response_model=InferenceResult)
def get_latest_inference() -> InferenceResult:
    latest = prediction_buffer.latest()
    if not latest:
        raise HTTPException(status_code=404, detail="No historical predictions present in the live memory stack.")
    return latest

@router.get("/history", response_model=InferenceHistory)
def get_inference_history() -> InferenceHistory:
    history_list = prediction_buffer.history()
    return InferenceHistory(count=len(history_list), history=history_list)

@router.get("/statistics", response_model=InferenceStatistics)
def get_inference_statistics() -> InferenceStatistics:
    return prediction_buffer.statistics()

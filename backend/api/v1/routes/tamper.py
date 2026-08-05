"""
REST Endpoints for the final Tamper Logic orchestration pipeline.
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from backend.schemas.tamper_schema import TamperEventResponse, TamperStatisticsResponse
from backend.tamper.tamper_engine import tamper_engine
from backend.tamper.history import tamper_history
from backend.inference.inference_engine import inference_engine
from backend.deviation.deviation_engine import deviation_engine
from backend.stream.camera_manager import CameraManager
from backend.config.logging import logger

router = APIRouter(prefix="/tamper", tags=["Tamper Logic Engine"])

@router.post("/evaluate", response_model=TamperEventResponse)
def evaluate_tamper() -> TamperEventResponse:
    """Executes the full chain synchronously: Buffer -> Inference -> Deviation -> Tamper Logic."""
    cam = CameraManager(camera_id="default")
    frames = cam.buffer.frames()
    
    if len(frames) < 15:
        raise HTTPException(status_code=400, detail="Insufficient physical frame buffer depth. Wait for hardware initialization.")
    
    try:
        # Phase 3: Inference Extraction (Reuses extraction internally)
        inf_res = inference_engine.run(frames[-15:])
        # Phase 4C: Deviation Calculation (Reuses features from inf_res)
        dev_res = deviation_engine.evaluate(inf_res.feature_vector)
        # Phase 4D: Physical Classification
        tamper_event = tamper_engine.evaluate(inf_res, dev_res)
        
        tamper_history.push(tamper_event)
        return TamperEventResponse(**tamper_event.__dict__)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Critical execution fault across evaluation chain: {e}")
        raise HTTPException(status_code=500, detail="Multistage logic engine execution fault occurred.")

@router.get("/latest", response_model=TamperEventResponse)
def get_latest_tamper() -> TamperEventResponse:
    """Retrieves the absolute most recent physical classification event."""
    event = tamper_history.latest()
    if not event:
        raise HTTPException(status_code=404, detail="No historical tamper events compiled inside execution arrays yet.")
    return TamperEventResponse(**event.__dict__)

@router.get("/history", response_model=List[TamperEventResponse])
def get_tamper_history() -> List[TamperEventResponse]:
    """Retrieves the rolling logical window buffer history tracks."""
    records = tamper_history.history()
    return [TamperEventResponse(**r.__dict__) for r in records]

@router.get("/statistics", response_model=TamperStatisticsResponse)
def get_tamper_statistics() -> TamperStatisticsResponse:
    """Generates composite analytical tracking distribution ratios."""
    return TamperStatisticsResponse(**tamper_history.statistics())

@router.delete("/history")
def clear_tamper_history() -> Dict[str, bool]:
    """Force flush all operational metric matrices stacks securely."""
    tamper_history.clear()
    return {"success": True}

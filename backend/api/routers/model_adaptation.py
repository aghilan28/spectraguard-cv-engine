from fastapi import APIRouter, HTTPException
from backend.services.model_adaptation_service import ModelAdaptationService

router = APIRouter(prefix="/api/v1/training", tags=["Model Adaptation & Certification"])
trainer = ModelAdaptationService()

@router.post("/start")
async def start_training():
    status = trainer.status
    if status["status"] == "running":
        return {"message": "GridSearchCV Training pipeline execution currently running.", "status": status}
    trainer.run_training_async()
    return {"message": "Background adaptation sequence triggered.", "status": trainer.status}

@router.get("/status")
async def get_training_status():
    return trainer.status

@router.get("/results")
async def get_training_results():
    return trainer.status

@router.get("/best-model")
async def get_best_model_info():
    return {"best_hyperparameters": trainer.status.get("best_params"), "metrics": trainer.status}

@router.post("/promote")
async def manual_promote_route():
    return trainer.promote_manually()

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
from backend.services.model_validation_service import ModelValidationService

router = APIRouter(prefix="/api/v1/model", tags=["Model Validation"])
validator = ModelValidationService()

@router.post("/validate")
async def start_validation():
    status = validator.get_status()
    if status["status"] == "running":
        return {"message": "Validation engine is already actively processing a workload.", "status": status}
    validator.run_validation_async()
    return {"message": "Validation sequence triggered successfully.", "status": validator.get_status()}

@router.get("/status")
async def get_validation_status():
    return validator.get_status()

@router.get("/results")
async def get_validation_results():
    return validator.get_summary()

@router.get("/report")
async def get_validation_report():
    csv_path = os.path.abspath("storage/reports/model_validation.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="The validation CSV report artifact has not been generated.")
    return {"report_path": csv_path}

@router.get("/errors")
async def get_validation_errors():
    return validator.get_errors()

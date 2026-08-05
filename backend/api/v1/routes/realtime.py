from fastapi import APIRouter, HTTPException
from backend.realtime.realtime_engine import realtime_engine

# The APIRouter object MUST be named 'router'
router = APIRouter(prefix="/realtime", tags=["Realtime Engine"])

@router.post("/start")
def start_realtime():
    if realtime_engine.start():
        return {"status": "started", "message": "Realtime orchestration engine initialized."}
    raise HTTPException(status_code=409, detail="Engine is already running.")

@router.post("/stop")
def stop_realtime():
    if realtime_engine.stop():
        return {"status": "stopped", "message": "Realtime orchestration engine halted."}
    return {"status": "idle", "message": "Engine was not running."}

@router.post("/pause")
def pause_realtime():
    if realtime_engine.pause():
        return {"status": "paused", "message": "Orchestration cycle suspended."}
    raise HTTPException(status_code=400, detail="Cannot pause. Engine is not running or already paused.")

@router.post("/resume")
def resume_realtime():
    if realtime_engine.resume():
        return {"status": "resumed", "message": "Orchestration cycle resumed."}
    raise HTTPException(status_code=400, detail="Cannot resume. Engine is not paused.")

@router.get("/status")
def get_status():
    with realtime_engine.state._lock:
        st = realtime_engine.state
        uptime = st.get_uptime()
        fps = st.processed_count / uptime if uptime > 0 else 0.0
        return {
            "running": st.running,
            "paused": st.paused,
            "processed_count": st.processed_count,
            "fps": round(fps, 2),
            "uptime_seconds": round(uptime, 2)
        }

@router.get("/latest")
def get_latest():
    record = realtime_engine.history.get_latest()
    if not record:
        raise HTTPException(status_code=404, detail="No processing records available.")
    return record

@router.get("/history")
def get_history():
    return realtime_engine.history.get_all()

@router.get("/statistics")
def get_statistics():
    return realtime_engine.metrics.get_statistics(uptime=realtime_engine.state.get_uptime())

@router.delete("/history")
def clear_history():
    realtime_engine.history.clear()
    return {"status": "cleared", "message": "Rolling history buffer purged."}

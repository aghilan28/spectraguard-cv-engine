import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.config.logging import logger
from backend.stream.camera_manager import CameraManager
from backend.utils.image_utils import preprocess_frame, encode_jpeg

router = APIRouter(tags=["WebSocket Engine"])

class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Live broadcast WebSocket connection opened. Active clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client safely logged off. Active broadcast targets left: {len(self.active_connections)}")

manager = ConnectionManager()

@router.websocket("/ws/live")
async def live_stream_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    camera = CameraManager(camera_id="default")
    
    try:
        while True:
            # Periodic check context to process incoming control messages or catch connection drops
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
            except asyncio.TimeoutError:
                pass
                
            frame = camera.get_latest_frame()
            if frame is not None:
                try:
                    processed = preprocess_frame(frame, target_size=None, add_timestamp=True)
                    jpeg_bytes = encode_jpeg(processed, quality=75)
                    await websocket.send_bytes(jpeg_bytes)
                except Exception as stream_err:
                    logger.warning(f"WebSocket processing loop runtime warning: {stream_err}")
                    
            # Yield synchronization bounds matching execution speeds (~30 FPS targets)
            await asyncio.sleep(0.033)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as general_err:
        logger.error(f"Fatal connection exception caught inside socket stack: {general_err}")
        manager.disconnect(websocket)

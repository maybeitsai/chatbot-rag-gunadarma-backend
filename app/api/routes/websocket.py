"""
Simple WebSocket route untuk testing dan Socket.IO placeholder
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Simple WebSocket endpoint for testing"""
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received: {data}")
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


@router.get("/ws/test")
async def test_ws_endpoint():
    """Test endpoint to verify /ws path works"""
    return {"message": "WebSocket path is working"}


@router.get("/ws/socket.io/")
async def socket_io_placeholder():
    """Placeholder untuk Socket.IO requests - menghindari 404 error"""
    return JSONResponse(
        status_code=200,
        content={"message": "Socket.IO not implemented yet", "status": "placeholder"}
    )


@router.post("/ws/socket.io/")
async def socket_io_placeholder_post():
    """Placeholder untuk Socket.IO POST requests"""
    return JSONResponse(
        status_code=200,
        content={"message": "Socket.IO not implemented yet", "status": "placeholder"}
    )

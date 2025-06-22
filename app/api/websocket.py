"""
WebSocket dan Socket.IO configuration untuk real-time communication
"""
import logging
import socketio
from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=False
)

# Create Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio)


@sio.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    logger.info(f"Socket.IO client {sid} connected")
    await sio.emit('connection_status', {'status': 'connected'}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Socket.IO client {sid} disconnected")


@sio.event
async def ask_question(sid, data):
    """Handle question from client via Socket.IO"""
    try:
        question = data.get('question', '') if isinstance(data, dict) else str(data)
        
        if not question:
            await sio.emit('error', {'message': 'Question is required'}, room=sid)
            return
        
        logger.info(f"Socket.IO question from {sid}: {question}")
        
        # Emit processing status
        await sio.emit('processing', {'status': 'Processing your question...'}, room=sid)
        
        # Get RAG pipeline from app state
        from app.api.dependencies import AppState
        app_state = AppState()
        rag_pipeline = app_state.get_rag_pipeline()
        
        if not rag_pipeline:
            await sio.emit('error', {'message': 'RAG system not available'}, room=sid)
            return
        
        # Process question
        try:
            if hasattr(rag_pipeline, 'ask_async'):
                response = await rag_pipeline.ask_async(question)
            else:
                response = rag_pipeline.ask(question)
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            await sio.emit('error', {'message': 'Failed to get answer from RAG system'}, room=sid)
            return
        
        # Send response
        result = {
            'question': question,
            'answer': response.get('answer', '') if isinstance(response, dict) else str(response),
            'sources': response.get('sources', []) if isinstance(response, dict) else [],
            'processing_time': response.get('processing_time', 0) if isinstance(response, dict) else 0
        }
        
        await sio.emit('answer', result, room=sid)
        logger.info(f"Socket.IO response sent to {sid}")
        
    except Exception as e:
        logger.error(f"Error processing question via Socket.IO: {e}")
        await sio.emit('error', {'message': 'Failed to process question'}, room=sid)


def setup_socketio(app: FastAPI):
    """Setup Socket.IO with FastAPI app"""
    # Mount Socket.IO pada path /ws
    app.mount("/ws", socket_app)
    logger.info("Socket.IO mounted at /ws/")
    return app

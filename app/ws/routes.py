"""WebSocket routes for real-time pipeline progress and system events."""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ws.manager import manager

logger = logging.getLogger(__name__)
ws_router = APIRouter(tags=["websocket"])


@ws_router.websocket("/ws/pipeline/{batch_id}")
async def ws_pipeline(ws: WebSocket, batch_id: str):
    """Subscribe to real-time progress for a specific pipeline run."""
    await manager.connect(ws, batch_id=batch_id)
    try:
        while True:
            # Keep connection alive; client can send pings
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws, batch_id=batch_id)
        logger.info("WS disconnected from batch %s", batch_id)


@ws_router.websocket("/ws/global")
async def ws_global(ws: WebSocket):
    """Subscribe to system-wide events (watchdog detections, alerts)."""
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
        logger.info("Global WS disconnected")

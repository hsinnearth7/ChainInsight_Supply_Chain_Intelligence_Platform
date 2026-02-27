"""WebSocket routes for real-time pipeline progress and system events."""

import hmac
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import API_KEY
from app.ws.manager import manager

logger = logging.getLogger(__name__)
ws_router = APIRouter(tags=["websocket"])


def _check_ws_api_key(api_key: str | None) -> bool:
    """Validate WebSocket API key (passed as query param)."""
    if api_key is None:
        return False
    return hmac.compare_digest(api_key, API_KEY)


@ws_router.websocket("/ws/pipeline/{batch_id}")
async def ws_pipeline(ws: WebSocket, batch_id: str, api_key: str = Query(None)):
    """Subscribe to real-time progress for a specific pipeline run."""
    if not _check_ws_api_key(api_key):
        await ws.close(code=4003, reason="Invalid or missing API key")
        return
    await manager.connect(ws, batch_id=batch_id)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws, batch_id=batch_id)
        logger.info("WS disconnected from batch %s", batch_id)


@ws_router.websocket("/ws/global")
async def ws_global(ws: WebSocket, api_key: str = Query(None)):
    """Subscribe to system-wide events (watchdog detections, alerts)."""
    if not _check_ws_api_key(api_key):
        await ws.close(code=4003, reason="Invalid or missing API key")
        return
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
        logger.info("Global WS disconnected")

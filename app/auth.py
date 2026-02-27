"""API Key authentication for ChainInsight."""

import hmac
import logging

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import API_KEY

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(_api_key_header)) -> str:
    """FastAPI dependency that validates the X-API-Key header.

    Exempt endpoints (like /api/health) should NOT include this dependency.
    """
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if not hmac.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

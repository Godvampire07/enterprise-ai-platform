import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from backend.app.core.logging import logger

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Log request details
        logger.info(f"RID: {request_id} - {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        # Log response details
        logger.info(
            f"RID: {request_id} - Status: {response.status_code} - Time: {process_time:.4f}s"
        )
        
        return response

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.api.v1 import auth, users, health, documents
from backend.app.middleware.logging import LoggingMiddleware
from backend.app.core.exceptions import BaseAppException
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting Enterprise AI Platform Backend...")
    yield
    # Shutdown logic
    logger.info("Shutting down Enterprise AI Platform Backend...")
 
def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
 
    # Set CORS
    if settings.BACKEND_CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
 
    # Add Custom Middlewares
    application.add_middleware(LoggingMiddleware)
 
    # Include Routers
    application.include_router(health.router, prefix=settings.API_V1_STR)
    application.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
    application.include_router(users.router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])
    application.include_router(documents.router, prefix=f"{settings.API_V1_STR}/documents", tags=["documents"])

    # Global Exception Handlers
    @application.exception_handler(BaseAppException)
    async def app_exception_handler(request: Request, exc: BaseAppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": getattr(request.state, "request_id", None)},
        )

    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal Server Error",
                "request_id": getattr(request.state, "request_id", None)
            },
        )

    return application

app = create_application()

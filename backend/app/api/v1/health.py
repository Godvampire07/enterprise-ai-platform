from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class HealthCheck(BaseModel):
    status: str = "ok"

@router.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    return HealthCheck(status="ok")

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.database.redis import get_redis, RedisClient
from backend.app.repositories.user_repository import UserRepository
from backend.app.services.auth_service import AuthService
from backend.app.schemas.auth import Token, RefreshTokenRequest
from backend.app.schemas.user import UserCreate, User as UserSchema
from backend.app.dependencies.auth import get_current_user
from backend.app.models.user import User

router = APIRouter()

@router.post("/register", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
def register(
    *,
    db: Session = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
    user_in: UserCreate
):
    user_repo = UserRepository(db)
    auth_service = AuthService(user_repo, redis)
    return auth_service.register(user_in)

@router.post("/login", response_model=Token)
def login(
    *,
    db: Session = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
    form_data: OAuth2PasswordRequestForm = Depends()
):
    user_repo = UserRepository(db)
    auth_service = AuthService(user_repo, redis)
    return auth_service.authenticate_user(form_data.username, form_data.password)

@router.post("/refresh", response_model=Token)
def refresh_token(
    *,
    db: Session = Depends(get_db),
    redis: RedisClient = Depends(get_redis),
    token_request: RefreshTokenRequest
):
    user_repo = UserRepository(db)
    auth_service = AuthService(user_repo, redis)
    return auth_service.refresh_token(token_request.refresh_token)

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    *,
    redis: RedisClient = Depends(get_redis),
    current_user: User = Depends(get_current_user)
):
    auth_service = AuthService(None, redis)
    auth_service.logout(current_user.id)

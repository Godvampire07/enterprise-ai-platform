from datetime import timedelta
from typing import Optional
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_password,
    decode_token,
)
from backend.app.core.config import settings
from backend.app.core.exceptions import AuthenticationError
from backend.app.repositories.user_repository import UserRepository
from backend.app.schemas.auth import Token
from backend.app.database.redis import RedisClient

class AuthService:
    def __init__(self, user_repo: UserRepository, redis: RedisClient):
        self.user_repo = user_repo
        self.redis = redis

    def authenticate_user(self, username: str, password: str) -> Token:
        user = self.user_repo.get_by_username(username)
        if not user or not verify_password(password, user.hashed_password):
            raise AuthenticationError("Incorrect username or password")
        
        if not user.is_active:
            raise AuthenticationError("Inactive user")

        return self._generate_tokens(user.id)

    def refresh_token(self, refresh_token: str) -> Token:
        payload = decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise AuthenticationError("Invalid refresh token")
        
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Invalid token payload")

        # Check if refresh token is in Redis (whitelist/blacklist logic)
        cached_token = self.redis.get(f"refresh_token:{user_id}")
        if not cached_token or cached_token != refresh_token:
            raise AuthenticationError("Refresh token expired or revoked")

        user = self.user_repo.get_by_id(int(user_id))
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")

        return self._generate_tokens(user.id)

    def _generate_tokens(self, user_id: int) -> Token:
        access_token = create_access_token(subject=user_id)
        refresh_token = create_refresh_token(subject=user_id)

        # Store refresh token in Redis
        self.redis.set(
            f"refresh_token:{user_id}",
            refresh_token,
            expire=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )

    def logout(self, user_id: int):
        self.redis.delete(f"refresh_token:{user_id}")
    
    def register(self, user_in: "UserCreate") -> "User":
        if self.user_repo.get_by_email(user_in.email):
            from backend.app.core.exceptions import ConflictError
            raise ConflictError("User with this email already exists")
        if self.user_repo.get_by_username(user_in.username):
            from backend.app.core.exceptions import ConflictError
            raise ConflictError("User with this username already exists")
        
        return self.user_repo.create(user_in)

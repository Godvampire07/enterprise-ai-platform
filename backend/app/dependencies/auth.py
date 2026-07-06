from fastapi import Depends, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from sqlalchemy.orm import Session
from backend.app.core.security import decode_token
from backend.app.core.exceptions import AuthenticationError, ForbiddenError
from backend.app.database.session import get_db
from backend.app.models.user import User
from backend.app.repositories.user_repository import UserRepository
from backend.app.core.config import settings

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)

def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(reusable_oauth2)
) -> User:
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise AuthenticationError("Invalid access token")
    
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Token subject missing")
    
    user_repo = UserRepository(db)
    user = user_repo.get_by_id(int(user_id))
    if not user:
        raise AuthenticationError("User not found")
    if not user.is_active:
        raise AuthenticationError("User is inactive")
    
    return user

class RoleChecker:
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_user)):
        if user.role not in self.allowed_roles:
            raise ForbiddenError(f"Role {user.role} is not authorized")
        return user

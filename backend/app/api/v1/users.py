from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from backend.app.database.session import get_db
from backend.app.repositories.user_repository import UserRepository
from backend.app.services.user_service import UserService
from backend.app.schemas.user import User as UserSchema, UserUpdate
from backend.app.dependencies.auth import get_current_user, RoleChecker
from backend.app.models.user import User

router = APIRouter()

@router.get("/me", response_model=UserSchema)
def read_user_me(
    current_user: User = Depends(get_current_user)
):
    return current_user

@router.get("/", response_model=List[UserSchema], dependencies=[Depends(RoleChecker(["admin"]))])
def read_users(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    user_repo = UserRepository(db)
    user_service = UserService(user_repo)
    return user_service.get_users(skip=skip, limit=limit)

@router.patch("/me", response_model=UserSchema)
def update_user_me(
    *,
    db: Session = Depends(get_db),
    user_in: UserUpdate,
    current_user: User = Depends(get_current_user)
):
    user_repo = UserRepository(db)
    user_service = UserService(user_repo)
    return user_service.update_user(current_user.id, user_in)

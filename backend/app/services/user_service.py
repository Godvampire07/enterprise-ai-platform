from typing import List, Optional
from backend.app.repositories.user_repository import UserRepository
from backend.app.schemas.user import UserCreate, UserUpdate
from backend.app.models.user import User

class UserService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def get_user(self, user_id: int) -> Optional[User]:
        return self.user_repo.get_by_id(user_id)

    def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.user_repo.get_multi(skip=skip, limit=limit)

    def create_user(self, user_in: UserCreate) -> User:
        return self.user_repo.create(user_in)

    def update_user(self, user_id: int, user_in: UserUpdate) -> Optional[User]:
        user = self.user_repo.get_by_id(user_id)
        if not user:
            return None
        return self.user_repo.update(user, user_in)

    def delete_user(self, user_id: int) -> Optional[User]:
        return self.user_repo.delete(user_id)

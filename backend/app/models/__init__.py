from backend.app.database.base import Base
from backend.app.models.user import User

# Add all models here for Alembic discovery
__all__ = ["Base", "User"]

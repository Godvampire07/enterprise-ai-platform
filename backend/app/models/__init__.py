from backend.app.database.base import Base
from backend.app.models.user import User
from backend.app.models.document import Document
from backend.app.models.document_chunk import DocumentChunk

# Add all models here for Alembic discovery
__all__ = ["Base", "User", "Document", "DocumentChunk"]

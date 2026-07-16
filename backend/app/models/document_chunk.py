from sqlalchemy import Column, Integer, Text, ForeignKey, DateTime, func, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from backend.app.database.base import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)

    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)

    chunk_metadata = Column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        server_default="{}",
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return (
            f"<DocumentChunk(id={self.id}, "
            f"document_id={self.document_id}, "
            f"index={self.chunk_index}, "
            f"page={self.page_number})>"
        )
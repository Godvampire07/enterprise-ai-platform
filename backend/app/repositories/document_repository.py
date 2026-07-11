from typing import Optional, List
from sqlalchemy.orm import Session
from backend.app.models.document import Document

class DocumentRepository:
    def __init__(self, db: Session) -> None:
         self.db = db

    def create_document(
        self, 
        filename: str, 
        original_filename: str, 
        file_path: str, 
        file_size: int, 
        content_type: str
    ) -> Document:
        """Create document database row, flushing is done to acquire the ID
        without committing the transaction.
        """
        db_obj = Document(
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            content_type=content_type
        )
        self.db.add(db_obj)
        self.db.flush()
        return db_obj

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Fetch document metadata by ID."""
        return self.db.query(Document).filter(Document.id == doc_id).first()

    def get_all_documents(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """Fetch multiple document records with pagination limits."""
        return self.db.query(Document).offset(skip).limit(limit).all()

    def delete_document(self, doc_id: int) -> Optional[Document]:
        """Delete document metadata. Cascade rule handles chunk deletion."""
        obj = self.get_document(doc_id)
        if obj:
            self.db.delete(obj)
            self.db.flush()
        return obj

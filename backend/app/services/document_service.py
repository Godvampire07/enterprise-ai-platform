import os
from typing import Optional, List
from sqlalchemy.orm import Session
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.models.document import Document

class DocumentService:
    def __init__(self, db: Session, doc_repo: DocumentRepository) -> None:
        self.db = db
        self.doc_repo = doc_repo

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Fetch metadata for a specific document."""
        return self.doc_repo.get_document(doc_id)

    def get_all_documents(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Document]:
        """Fetch multiple document records with pagination limits."""
        return self.doc_repo.get_all_documents(user_id=user_id, skip=skip, limit=limit)

    def delete_document(self, doc_id: int) -> Optional[Document]:
        """Delete document record and associated file copy from disk."""
        doc = self.doc_repo.get_document(doc_id)
        if not doc:
            return None

        file_path = doc.file_path

        try:
            deleted_doc = self.doc_repo.delete_document(doc_id)
            self.db.commit()

            # Purge stored PDF from local disk
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    # Log error, but don't fail transaction rollback as DB record is already committed
                    pass

            return deleted_doc
        except Exception as e:
            self.db.rollback()
            raise e

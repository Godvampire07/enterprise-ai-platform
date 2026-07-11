import os
import uuid
import shutil
from typing import BinaryIO
from sqlalchemy.orm import Session
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.chunk_repository import ChunkRepository
from backend.app.services.parser_service import ParserService
from backend.app.services.chunking_service import ChunkingService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.models.document import Document

class DocumentIngestionService:
    def __init__(
        self,
        db: Session,
        doc_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        parser_service: ParserService,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService
    ) -> None:
        self.db = db
        self.doc_repo = doc_repo
        self.chunk_repo = chunk_repo
        self.parser_service = parser_service
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.upload_dir = "storage/uploads"
        os.makedirs(self.upload_dir, exist_ok=True)

    def ingest_document(
        self,
        file_content: BinaryIO,
        original_filename: str,
        content_type: str,
        file_size: int
    ) -> Document:
        """Runs the complete ingestion pipeline:
        Save file -> Create doc record -> Parse pages -> Chunk -> Embed -> Bulk Insert.
        Performs session rollback and deletes saved local files if any error occurs.
        """
        # Generate unique local filename
        file_ext = os.path.splitext(original_filename)[1]
        filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(self.upload_dir, filename)

        # 1. Save uploaded file to disk
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file_content, f)
        except Exception as e:
            raise RuntimeError(f"Failed to write file to local disk: {str(e)}")

        # 2. Database Transaction
        try:
            # Create Document database record
            doc = self.doc_repo.create_document(
                filename=filename,
                original_filename=original_filename,
                file_path=file_path,
                file_size=file_size,
                content_type=content_type
            )

            # Parse document pages
            pages = self.parser_service.parse_file(file_path, content_type)

            # Chunk document pages
            chunks = self.chunking_service.chunk_document(pages)

            # Generate Embeddings & Formulate Chunk Database Records
            if chunks:
                texts = [c["text"] for c in chunks]
                embeddings = self.embedding_service.embed_documents(texts)

                chunks_to_insert = []
                for i, c in enumerate(chunks):
                    chunks_to_insert.append({
                        "document_id": doc.id,
                        "chunk_index": c["chunk_index"],
                        "page_number": c["page_number"],
                        "text": c["text"],
                        "embedding": embeddings[i],
                        "metadata": {
                            "page": c["page_number"],
                            "source": original_filename
                        }
                    })

                # Bulk insert mapped chunks
                self.chunk_repo.insert_chunks(chunks_to_insert)

            # Commit the complete transaction
            self.db.commit()
            self.db.refresh(doc)
            return doc

        except Exception as e:
            self.db.rollback()
            # Clean up the file from disk if created
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            raise e

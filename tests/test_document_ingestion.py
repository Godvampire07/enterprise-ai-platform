import io
import os
import pytest
from unittest.mock import MagicMock
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import fitz

from backend.app.core.config import settings
from backend.app.database.base import Base
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.chunk_repository import ChunkRepository
from backend.app.models.document import Document
from backend.app.services.parser_service import ParserService
from backend.app.services.chunking_service import ChunkingService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.document_ingestion_service import DocumentIngestionService
from backend.app.services.document_service import DocumentService

@pytest.fixture(scope="module")
def db_session():
    db_url = str(settings.DATABASE_URL)
    if not db_url.startswith("postgresql"):
        pytest.skip("PostgreSQL environment is required for testing transaction rollbacks.")
        
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.fixture
def mock_embedding_service():
    service = EmbeddingService()
    # Mock model validation loading steps to prevent heavy sentence-transformer downloads
    service._ensure_model_loaded = MagicMock()
    service.embed_text = MagicMock(return_value=[0.1] * 384)
    service.embed_documents = MagicMock(side_effect=lambda doc_list: [[0.1] * 384 for _ in doc_list])
    return service

@pytest.fixture
def test_pdf_bytes():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Mock PDF document text content for ingestion pipeline.")
    pdf_io = io.BytesIO()
    doc.save(pdf_io)
    doc.close()
    pdf_io.seek(0)
    return pdf_io

def test_document_ingestion_success(db_session, mock_embedding_service, test_pdf_bytes):
    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)
    parser_service = ParserService()
    chunking_service = ChunkingService()
    
    ingestion_service = DocumentIngestionService(
        db=db_session,
        doc_repo=doc_repo,
        chunk_repo=chunk_repo,
        parser_service=parser_service,
        chunking_service=chunking_service,
        embedding_service=mock_embedding_service
    )
    
    # Ingest document
    doc = ingestion_service.ingest_document(
        file_content=test_pdf_bytes,
        original_filename="sample_ingest.pdf",
        content_type="application/pdf",
        file_size=1024
    )
    
    # Assert document details
    assert doc.id is not None
    assert doc.original_filename == "sample_ingest.pdf"
    assert os.path.exists(doc.file_path)
    
    # Validate DB state (chunks inserted)
    chunks = chunk_repo.get_chunks(doc.id)
    assert len(chunks) == 1
    assert chunks[0].text == "Mock PDF document text content for ingestion pipeline."
    assert chunks[0].embedding == [0.1] * 384
    
    # Clean up document using DocumentService CRUD manager
    doc_service = DocumentService(db_session, doc_repo)
    file_path = doc.file_path
    doc_service.delete_document(doc.id)
    
    # Ensure physical file is deleted
    assert not os.path.exists(file_path)

def test_document_ingestion_rollback_on_failure(db_session, mock_embedding_service, test_pdf_bytes):
    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)
    parser_service = ParserService()
    chunking_service = ChunkingService()
    
    # Mock embedding error to trigger rollback mechanism
    mock_embedding_service.embed_documents = MagicMock(side_effect=RuntimeError("Embedding generation failed."))
    
    ingestion_service = DocumentIngestionService(
        db=db_session,
        doc_repo=doc_repo,
        chunk_repo=chunk_repo,
        parser_service=parser_service,
        chunking_service=chunking_service,
        embedding_service=mock_embedding_service
    )
    
    # Ingest document (expect crash)
    with pytest.raises(RuntimeError) as excinfo:
        ingestion_service.ingest_document(
            file_content=test_pdf_bytes,
            original_filename="fail_ingest.pdf",
            content_type="application/pdf",
            file_size=1024
        )
    assert "Embedding generation failed." in str(excinfo.value)
    
    # Verify no document or chunks remain in DB
    db_doc = db_session.query(Document).filter(Document.original_filename == "fail_ingest.pdf").first()
    assert db_doc is None

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from backend.app.core.config import settings
from backend.app.database.base import Base
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.chunk_repository import ChunkRepository

@pytest.fixture(scope="module")
def db_session():
    db_url = str(settings.DATABASE_URL)
    if not db_url.startswith("postgresql"):
        pytest.skip("PostgreSQL environment is required for real repository tests.")
    
    engine = create_engine(db_url)
    
    # Ensure pgvector extension exists in the connected Postgres instance
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    
    # Create test user to satisfy foreign key constraints
    from backend.app.models.user import User
    test_user = User(id=1, username="testuser", email="test@example.com", hashed_password="mockedpassword")
    session.add(test_user)
    session.commit()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()

def test_repository_insertion_and_deletion(db_session):
    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)
    
    # 1. Create document Database row
    doc = doc_repo.create_document(
        filename="test_repo_file.pdf",
        original_filename="original.pdf",
        file_path="/storage/test_repo_file.pdf",
        file_size=1024,
        content_type="application/pdf",
        user_id=1
    )
    db_session.commit()
    
    # 2. Insert Chunks via Bulk Mapper
    chunks_data = [
        {
            "document_id": doc.id,
            "chunk_index": 0,
            "page_number": 1,
            "text": "This is chunk index zero.",
            "embedding": [0.1] * 384,
            "metadata": {"source": "original.pdf"}
        },
        {
            "document_id": doc.id,
            "chunk_index": 1,
            "page_number": 2,
            "text": "This is chunk index one.",
            "embedding": [0.2] * 384,
            "metadata": {"source": "original.pdf"}
        }
    ]
    chunk_repo.insert_chunks(chunks_data)
    
    # 3. Verify get_chunks paginations / sequences
    fetched = chunk_repo.get_chunks(doc.id)
    assert len(fetched) == 2
    assert fetched[0].text == "This is chunk index zero."
    assert fetched[0].chunk_index == 0
    assert fetched[1].text == "This is chunk index one."
    assert fetched[1].chunk_index == 1
    
    # 4. Verify cascade deletion deletes chunks
    doc_repo.delete_document(doc.id)
    db_session.commit()
    
    fetched_after_delete = chunk_repo.get_chunks(doc.id)
    assert len(fetched_after_delete) == 0

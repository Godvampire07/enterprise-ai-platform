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
        pytest.skip("PostgreSQL environment is required for testing pgvector similarity operations.")
        
    engine = create_engine(db_url)
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

def test_vector_similarity_search_precision(db_session):
    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)
    
    # 1. Create two test document rows
    doc1 = doc_repo.create_document("doc1.pdf", "doc1.pdf", "/tmp/doc1.pdf", 100, "application/pdf", user_id=1)
    doc2 = doc_repo.create_document("doc2.pdf", "doc2.pdf", "/tmp/doc2.pdf", 100, "application/pdf", user_id=1)
    db_session.commit()
    
    # 2. Setup vectors representing semantic directions
    # Chunk A: pointing towards direction 0
    # Chunk B: pointing towards direction 1
    # Chunk C: matching both 0 and 1 equally (but in doc2)
    emb_a = [1.0] + [0.0] * 383
    emb_b = [0.0, 1.0] + [0.0] * 382
    emb_c = [0.5, 0.5] + [0.0] * 382
    
    chunks_data = [
        {"document_id": doc1.id, "chunk_index": 0, "page_number": 1, "text": "Deep search vector A", "embedding": emb_a, "metadata": {}},
        {"document_id": doc1.id, "chunk_index": 1, "page_number": 2, "text": "Deep search vector B", "embedding": emb_b, "metadata": {}},
        {"document_id": doc2.id, "chunk_index": 0, "page_number": 1, "text": "Deep search vector C", "embedding": emb_c, "metadata": {}}
    ]
    chunk_repo.insert_chunks(chunks_data)
    
    # Query vector heavily resembling emb_a
    query_emb = [0.9] + [0.1] * 383
    
    # 3. Verify ordering: A is closest, C is neutral, B is furthest
    results = chunk_repo.similarity_search(query_embedding=query_emb, top_k=3)
    assert len(results) == 3
    assert results[0][0].text == "Deep search vector A"
    assert results[1][0].text == "Deep search vector C"
    assert results[2][0].text == "Deep search vector B"
    
    # Similarity scores should be descending
    assert results[0][1] > results[1][1]
    assert results[1][1] > results[2][1]
    
    # 4. Verify Document Bound Filter
    doc2_results = chunk_repo.similarity_search(query_embedding=query_emb, document_id=doc2.id)
    assert len(doc2_results) == 1
    assert doc2_results[0][0].text == "Deep search vector C"
    
    # 5. Verify Min Score Threshold Filter
    high_threshold_results = chunk_repo.similarity_search(query_embedding=query_emb, min_score=0.8)
    assert len(high_threshold_results) == 1
    assert high_threshold_results[0][0].text == "Deep search vector A"
    
    # Cleanup
    doc_repo.delete_document(doc1.id)
    doc_repo.delete_document(doc2.id)
    db_session.commit()

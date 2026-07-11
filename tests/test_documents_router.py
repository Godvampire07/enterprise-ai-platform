import io
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import fitz

from backend.app.main import app
from backend.app.database.session import get_db
from backend.app.database.base import Base
from backend.app.core.config import settings
from backend.app.dependencies.auth import get_current_user
from backend.app.models.user import User
from backend.app.services.embedding_service import EmbeddingService

@pytest.fixture(scope="module")
def db_session():
    db_url = str(settings.DATABASE_URL)
    if not db_url.startswith("postgresql"):
        pytest.skip("PostgreSQL environment is required to test router endpoints.")
        
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

@pytest.fixture(scope="module")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
            
    # Mock Auth dependency to yield a mock User
    def override_get_current_user():
        return User(id=1, username="testadmin", email="admin@enterprise.ai", role="admin")

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    with TestClient(app) as c:
        yield c
        
    del app.dependency_overrides[get_db]
    del app.dependency_overrides[get_current_user]

@pytest.fixture(autouse=True)
def mock_embedding_service(monkeypatch):
    # Keep embeddings mocked during API calls to bypass downloads or CPU latency
    monkeypatch.setattr(EmbeddingService, "_ensure_model_loaded", MagicMock())
    monkeypatch.setattr(EmbeddingService, "embed_text", MagicMock(return_value=[0.1] * 384))
    monkeypatch.setattr(EmbeddingService, "embed_documents", MagicMock(side_effect=lambda doc_list: [[0.1] * 384 for _ in doc_list]))

def test_router_upload_retrieve_search_delete(client):
    # 1. Create PDF file bytes
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "FastAPI router integration test content.")
    pdf_bytes = io.BytesIO()
    doc.save(pdf_bytes)
    doc.close()
    pdf_bytes.seek(0)

    # 2. Test File Upload Endpoint (POST /upload)
    response = client.post(
        f"{settings.API_V1_STR}/documents/upload",
        files={"file": ("router_test.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 201
    doc_res = response.json()
    doc_id = doc_res["id"]
    assert doc_res["original_filename"] == "router_test.pdf"

    # 3. Test Metadata Fetch Endpoint (GET /{id})
    get_res = client.get(f"{settings.API_V1_STR}/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == doc_id

    # 4. Test Retrieve Chunks Endpoint (GET /{id}/chunks)
    chunks_res = client.get(f"{settings.API_V1_STR}/documents/{doc_id}/chunks")
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()
    assert len(chunks) == 1
    assert "FastAPI router integration" in chunks[0]["text"]

    # 5. Test Vector Similarity Search Endpoint (POST /search)
    search_res = client.post(
        f"{settings.API_V1_STR}/documents/search",
        json={"query": "router test", "document_id": doc_id, "top_k": 3}
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert "results" in search_data
    assert len(search_data["results"]) == 1
    assert search_data["results"][0]["chunk"]["document_id"] == doc_id
    assert search_data["results"][0]["similarity_score"] is not None

    # 6. Test Delete Endpoint (DELETE /{id})
    del_res = client.delete(f"{settings.API_V1_STR}/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["id"] == doc_id

    # 7. Verify deletion cascade
    get_res_gone = client.get(f"{settings.API_V1_STR}/documents/{doc_id}")
    assert get_res_gone.status_code == 404

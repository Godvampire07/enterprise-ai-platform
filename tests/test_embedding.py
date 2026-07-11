from backend.app.services.embedding_service import EmbeddingService

def test_embedding_singleton_instance():
    service1 = EmbeddingService()
    service2 = EmbeddingService()
    # Verify that the underlying loaded transformer model is identical (singleton)
    assert service1._model is service2._model

def test_embedding_generation_dimensions():
    service = EmbeddingService()
    text = "Semantic text segment example."
    emb = service.embed_text(text)
    
    assert isinstance(emb, list)
    assert len(emb) == 384
    assert all(isinstance(val, float) for val in emb)

def test_embedding_bulk_documents():
    service = EmbeddingService()
    texts = ["Hello world of AI", "Deep learning embedding pipelines"]
    embs = service.embed_documents(texts)
    
    assert isinstance(embs, list)
    assert len(embs) == 2
    assert len(embs[0]) == 384
    assert len(embs[1]) == 384

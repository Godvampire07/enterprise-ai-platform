from backend.app.services.chunking_service import ChunkingService

def test_chunk_text():
    service = ChunkingService()
    # Create long text that exceeds chunk size limit (settings.CHUNK_SIZE=500)
    text = " ".join(["word"] * 200) # ~ 1000 characters
    chunks = service.chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= service.chunk_size

def test_chunk_page():
    service = ChunkingService()
    text = "Short text segment."
    page_number = 3
    page_chunks = service.chunk_page(text, page_number=page_number)
    
    assert len(page_chunks) == 1
    assert page_chunks[0]["text"] == text
    assert page_chunks[0]["page_number"] == page_number
    assert page_chunks[0]["chunk_index"] == 0

def test_chunk_document_ordered():
    service = ChunkingService()
    pages = [
        {"page": 1, "text": "First page text content."},
        {"page": 2, "text": "Second page text content."}
    ]
    doc_chunks = service.chunk_document(pages)
    
    assert len(doc_chunks) == 2
    assert doc_chunks[0]["text"] == "First page text content."
    assert doc_chunks[0]["page_number"] == 1
    assert doc_chunks[0]["chunk_index"] == 0
    
    assert doc_chunks[1]["text"] == "Second page text content."
    assert doc_chunks[1]["page_number"] == 2
    assert doc_chunks[1]["chunk_index"] == 1

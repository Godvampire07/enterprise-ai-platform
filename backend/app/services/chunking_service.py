from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.core.config import settings

class ChunkingService:
    def __init__(self) -> None:
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def chunk_text(self, text: str) -> List[str]:
        """Split a single string into a list of text chunks."""
        if not text.strip():
            return []
        return self.splitter.split_text(text)

    def chunk_page(self, text: str, page_number: int) -> List[Dict[str, Any]]:
        """Split a page's text, preserving the page number and returning dictionaries:
        [
            {
                "text": "...",
                "page_number": page_number,
                "chunk_index": i
            }
        ]
        """
        chunks = self.chunk_text(text)
        return [
            {
                "text": chunk,
                "page_number": page_number,
                "chunk_index": i
            }
            for i, chunk in enumerate(chunks)
        ]

    def chunk_document(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split an entire document's list of pages, aggregating chunks and indexing flat:
        [
            {
                "text": "...",
                "page_number": page_number,
                "chunk_index": global_index
            }
        ]
        """
        all_chunks = []
        global_index = 0
        for p in pages:
            page_num = p["page"]
            text = p["text"]
            page_chunks = self.chunk_page(text, page_num)
            for chunk in page_chunks:
                all_chunks.append({
                    "text": chunk["text"],
                    "page_number": page_num,
                    "chunk_index": global_index
                })
                global_index += 1
        return all_chunks

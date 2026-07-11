from typing import List, Optional, Tuple
from backend.app.repositories.chunk_repository import ChunkRepository
from backend.app.models.document_chunk import DocumentChunk
from backend.app.core.config import settings

class VectorService:
    def __init__(self, chunk_repo: ChunkRepository) -> None:
        self.chunk_repo = chunk_repo

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        document_id: Optional[int] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Handles vector similarity search orchestrations.
        Delegates vector search query to ChunkRepository, passing limits and filters.
        """
        if top_k is None:
            top_k = settings.TOP_K

        return self.chunk_repo.similarity_search(
            query_embedding=query_embedding,
            document_id=document_id,
            top_k=top_k,
            min_score=min_score
        )

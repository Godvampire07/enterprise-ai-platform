from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from backend.app.models.document_chunk import DocumentChunk

class ChunkRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def insert_chunks(self, chunks_data: List[dict]) -> None:
        """Optimized bulk insertion of mapping dictionaries."""
        if chunks_data:
            self.db.bulk_insert_mappings(DocumentChunk, chunks_data)
            self.db.flush()

    def get_chunks(self, document_id: int, skip: int = 0, limit: int = 100) -> List[DocumentChunk]:
        """Fetch chunks linked to a document ordered sequentially by index."""
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def delete_chunks(self, document_id: int) -> int:
        """Purge chunks linking to document ID manually."""
        deleted_count = (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .delete()
        )
        self.db.flush()
        return deleted_count

    def similarity_search(
        self,
        query_embedding: List[float],
        document_id: Optional[int] = None,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform vector similarity search using PGVector.
        Similarity score = 1.0 - Cosine Distance.
        """
        # cosine_distance maps directly to the <=> Operator in SQL
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        similarity = 1.0 - distance

        query = self.db.query(DocumentChunk, similarity.label("score"))
        
        if document_id is not None:
            query = query.filter(DocumentChunk.document_id == document_id)

        if min_score is not None:
            query = query.filter(similarity >= min_score)

        # Ordering by proximity distance ascending (which is similarity descending)
        results = query.order_by(distance).limit(top_k).all()
        return [(chunk, float(score)) for chunk, score in results]

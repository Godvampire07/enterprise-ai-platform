"""
Retriever Service.

Responsible for fetching relevant document chunks given a query embedding.
Extends beyond VectorService by supporting:
  - Multi-document filtering (search across specific document IDs)
  - Metadata enrichment (joins document name onto each result)
  - Score threshold enforcement

Composes ChunkRepository.similarity_search() and DocumentRepository —
no vector search logic is duplicated.

Position in the RAG pipeline:
  EmbeddingService → **RetrieverService** → PromptService → LLMProvider
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.app.core.config import settings
from backend.app.core.exceptions import RetrievalError, ForbiddenError, NotFoundError
from backend.app.core.logging import logger
from backend.app.models.document_chunk import DocumentChunk
from backend.app.repositories.chunk_repository import ChunkRepository
from backend.app.repositories.document_repository import DocumentRepository


@dataclass
class RetrievedChunk:
    """A chunk enriched with its parent document metadata.

    This is the domain object passed downstream to the PromptService,
    containing everything needed to both build context and cite sources.
    """

    text: str
    document_name: str
    document_id: int
    page_number: Optional[int]
    chunk_index: int
    similarity_score: float
    metadata: Dict


class RetrieverService:
    """Retrieves and enriches relevant chunks from pgvector."""

    def __init__(
        self,
        chunk_repo: ChunkRepository,
        doc_repo: DocumentRepository,
    ) -> None:
        self._chunk_repo = chunk_repo
        self._doc_repo = doc_repo

    def retrieve(
        self,
        query_embedding: List[float],
        user_id: int,
        document_ids: Optional[List[int]] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Execute vector similarity search and enrich results with document metadata.

        Args:
            query_embedding: The embedding vector of the user's question.
            user_id: The ID of the authenticated user to restrict access.
            document_ids: Optional list of document IDs to restrict the search.
                          When empty or None, all documents owned by the user are searched.
            top_k: Number of top results to retrieve per document scope.
            score_threshold: Minimum cosine similarity to include a result.

        Returns:
            List of RetrievedChunk objects sorted by similarity (descending).

        Raises:
            RetrievalError: On any database or search failure.
            NotFoundError: If a requested document does not exist.
            ForbiddenError: If a requested document does not belong to the user.
        """
        effective_top_k = top_k or settings.TOP_K
        effective_threshold = score_threshold or settings.RAG_SCORE_THRESHOLD

        # Resolve document boundaries and check ownership
        if document_ids:
            for doc_id in document_ids:
                doc = self._doc_repo.get_document(doc_id)
                if not doc:
                    raise NotFoundError(f"Document {doc_id} not found.")
                if doc.user_id != user_id:
                    raise ForbiddenError(f"You do not have access to document {doc_id}.")
        else:
            user_docs = self._doc_repo.get_all_documents(user_id=user_id)
            document_ids = [d.id for d in user_docs]
            if not document_ids:
                logger.info(f"Retriever found 0 documents for user {user_id}")
                return []

        logger.info(
            f"Retriever searching | user_id={user_id} | top_k={effective_top_k} | "
            f"threshold={effective_threshold} | "
            f"document_filter={document_ids}"
        )

        try:
            raw_results = self._search_chunks(
                query_embedding=query_embedding,
                document_ids=document_ids,
                top_k=effective_top_k,
                score_threshold=effective_threshold,
            )
        except Exception as e:
            if isinstance(e, (ForbiddenError, NotFoundError)):
                raise e
            logger.error(f"Vector search failed: {e}", exc_info=True)
            raise RetrievalError(f"Vector similarity search failed: {str(e)}")

        if not raw_results:
            logger.info("Retriever found 0 matching chunks")
            return []

        # Enrich with document metadata
        enriched = self._enrich_with_metadata(raw_results)

        logger.info(
            f"Retriever returning {len(enriched)} chunks | "
            f"score_range=[{enriched[-1].similarity_score:.3f}, "
            f"{enriched[0].similarity_score:.3f}]"
        )

        return enriched

    def _search_chunks(
        self,
        query_embedding: List[float],
        document_ids: Optional[List[int]],
        top_k: int,
        score_threshold: float,
    ) -> List[tuple]:
        """Run similarity search, optionally filtering by document IDs.

        When document_ids is provided, we run separate searches per document
        to ensure fair top-k representation across documents, then merge
        and re-sort. When empty, we search the entire corpus.
        """
        if not document_ids:
            # Search across all documents
            return self._chunk_repo.similarity_search(
                query_embedding=query_embedding,
                document_id=None,
                top_k=top_k,
                min_score=score_threshold,
            )

        # Multi-document search: query each document, merge results
        all_results = []
        for doc_id in document_ids:
            results = self._chunk_repo.similarity_search(
                query_embedding=query_embedding,
                document_id=doc_id,
                top_k=top_k,
                min_score=score_threshold,
            )
            all_results.extend(results)

        # Sort by similarity score descending and take top_k overall
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def _enrich_with_metadata(
        self,
        raw_results: List[tuple],
    ) -> List[RetrievedChunk]:
        """Join document names onto chunk results via a local cache.

        Uses a dict to avoid redundant DB queries when multiple chunks
        come from the same document.
        """
        doc_name_cache: Dict[int, str] = {}
        enriched: List[RetrievedChunk] = []

        for chunk, score in raw_results:
            doc_id = chunk.document_id

            # Cache document name lookup
            if doc_id not in doc_name_cache:
                doc = self._doc_repo.get_document(doc_id)
                doc_name_cache[doc_id] = (
                    doc.original_filename if doc else f"Document {doc_id}"
                )

            enriched.append(
                RetrievedChunk(
                    text=chunk.text,
                    document_name=doc_name_cache[doc_id],
                    document_id=doc_id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    similarity_score=float(score),
                    metadata=chunk.chunk_metadata or {},
                )
            )

        return enriched

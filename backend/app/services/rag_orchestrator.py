"""
RAG Orchestrator.

The single coordinator that wires the entire question-answering pipeline.
This service contains NO business logic — it delegates each step to its
respective service and assembles the final response.

Pipeline flow:
  question
    → EmbeddingService.embed_text()          (query vectorization)
    → RetrieverService.retrieve()            (cosine similarity search)
    → PromptService.build_prompt()           (grounded prompt construction)
    → LLMProvider.generate()                 (answer generation)
    → _build_response()                      (response + source assembly)

The API layer calls ONLY this orchestrator — never individual services.
"""

from typing import List, Optional

from backend.app.core.logging import logger
from backend.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    RetrievalMetadata,
    SourceReference,
)
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.llm.base import LLMProvider, LLMResponse
from backend.app.services.prompt_service import PromptService
from backend.app.services.retriever_service import RetrievedChunk, RetrieverService


class RAGOrchestrator:
    """Coordinates the full RAG pipeline from question to grounded answer.

    Follows composition over inheritance — each dependency is injected,
    making the orchestrator fully testable and swappable.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        retriever_service: RetrieverService,
        prompt_service: PromptService,
        llm_provider: LLMProvider,
    ) -> None:
        self._embedding = embedding_service
        self._retriever = retriever_service
        self._prompt = prompt_service
        self._llm = llm_provider

    async def process_question(self, request: ChatRequest, user_id: int) -> ChatResponse:
        """Execute the full RAG pipeline for a user question.

        Steps:
          1. Generate query embedding using the same model as indexing
          2. Retrieve top-k similar chunks from pgvector
          3. Build a grounded prompt with context + question
          4. Call the LLM for a contextual answer
          5. Assemble the response with source citations

        Args:
            request: The validated ChatRequest from the API layer.
            user_id: The authenticated user's ID.

        Returns:
            ChatResponse containing the answer, sources, and metadata.
        """
        logger.info(
            f"RAG pipeline started | user_id={user_id} | question_length={len(request.question)} | "
            f"document_filter={request.document_ids or 'all'}"
        )

        # Step 1: Generate query embedding
        query_embedding = self._generate_query_embedding(request.question)

        # Step 2: Retrieve relevant chunks
        chunks = self._retrieve_chunks(
            query_embedding=query_embedding,
            user_id=user_id,
            document_ids=request.document_ids or None,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )

        # Step 3: Build grounded prompt
        constructed_prompt = self._prompt.build_prompt(
            chunks=chunks,
            question=request.question,
        )

        # Step 4: Call LLM
        llm_response = await self._llm.generate(
            prompt=constructed_prompt.user_prompt,
            system_prompt=constructed_prompt.system_prompt,
        )

        # Step 5: Assemble response
        response = self._build_response(
            llm_response=llm_response,
            chunks=chunks,
            conversation_id=request.conversation_id,
        )

        logger.info(
            f"RAG pipeline complete | sources={len(response.sources)} | "
            f"model={response.model} | "
            f"tokens={llm_response.usage.get('total_tokens', 'N/A')}"
        )

        return response

    def _generate_query_embedding(self, question: str) -> List[float]:
        """Vectorize the user question using the same embedding model as indexing.

        This ensures query embeddings are in the same vector space as the
        stored document chunk embeddings (all-MiniLM-L6-v2, 384 dimensions).
        """
        logger.debug("Generating query embedding")
        embedding = self._embedding.embed_text(question)

        if not embedding:
            logger.warning("Empty embedding generated for question")

        return embedding

    def _retrieve_chunks(
        self,
        query_embedding: List[float],
        user_id: int,
        document_ids: Optional[List[int]],
        top_k: Optional[int],
        score_threshold: Optional[float],
    ) -> List[RetrievedChunk]:
        """Delegate chunk retrieval to RetrieverService."""
        return self._retriever.retrieve(
            query_embedding=query_embedding,
            user_id=user_id,
            document_ids=document_ids,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    def _build_response(
        self,
        llm_response: LLMResponse,
        chunks: List[RetrievedChunk],
        conversation_id: Optional[str],
    ) -> ChatResponse:
        """Assemble the ChatResponse from LLM output and retrieved chunks.

        Maps each retrieved chunk to a SourceReference citation and computes
        retrieval metadata for observability.
        """
        # Build source citations
        sources = [
            SourceReference(
                document_name=chunk.document_name,
                page=chunk.page_number,
                chunk_index=chunk.chunk_index,
                similarity_score=round(chunk.similarity_score, 4),
            )
            for chunk in chunks
        ]

        # Compute retrieval metadata
        avg_score = (
            sum(c.similarity_score for c in chunks) / len(chunks) if chunks else 0.0
        )
        unique_docs = len({c.document_id for c in chunks})

        retrieval_metadata = RetrievalMetadata(
            chunks_retrieved=len(chunks),
            avg_similarity_score=round(avg_score, 4),
            documents_searched=unique_docs,
        )

        return ChatResponse(
            answer=llm_response.content,
            sources=sources,
            conversation_id=conversation_id or ChatResponse.model_fields["conversation_id"].default_factory(),
            model=llm_response.model,
            retrieval_metadata=retrieval_metadata,
        )

"""
Pydantic schemas for the RAG Chat API.

These models define the contract for POST /chat — the request payload,
the structured answer response, and the source citation format.
"""

import uuid
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Inbound question payload from the client.

    Attributes:
        question: The natural-language question to answer.
        document_ids: Optional list of document IDs to restrict the search scope.
                      When empty, all indexed documents are searched.
        conversation_id: Optional ID for multi-turn context (reserved for Phase 2).
        top_k: Number of top chunks to retrieve. Defaults to server config.
        score_threshold: Minimum cosine similarity score to include a chunk.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask against indexed documents.",
    )
    document_ids: List[int] = Field(
        default_factory=list,
        description="Restrict search to these document IDs. Empty means search all.",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID for multi-turn context.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of top chunks to retrieve.",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold.",
    )


class SourceReference(BaseModel):
    """A single source citation linking an answer back to its origin chunk.

    Every answer includes one or more of these so the client can verify
    the grounding of the response.
    """

    document_name: str = Field(
        ..., description="Original filename of the source document."
    )
    page: Optional[int] = Field(
        None, description="Page number within the source document."
    )
    chunk_index: int = Field(
        ..., description="Sequential index of the chunk within the document."
    )
    similarity_score: float = Field(
        ..., description="Cosine similarity score between query and chunk."
    )


class RetrievalMetadata(BaseModel):
    """Metadata about the retrieval step — useful for debugging and observability."""

    chunks_retrieved: int = Field(
        ..., description="Number of chunks that passed the similarity threshold."
    )
    avg_similarity_score: float = Field(
        ..., description="Average similarity across retrieved chunks."
    )
    documents_searched: int = Field(
        ..., description="Number of distinct documents that contributed chunks."
    )


class ChatResponse(BaseModel):
    """Outbound response containing the grounded answer and its provenance.

    The response always includes the LLM-generated answer, the source citations,
    and metadata about both the retrieval and generation steps.
    """

    answer: str = Field(..., description="The LLM-generated answer grounded in context.")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Citations linking the answer to source document chunks.",
    )
    conversation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation ID for tracking multi-turn sessions.",
    )
    model: str = Field(..., description="The LLM model used for generation.")
    retrieval_metadata: RetrievalMetadata = Field(
        ..., description="Statistics about the retrieval step."
    )

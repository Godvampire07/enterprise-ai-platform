"""
Chat API Router.

Thin route layer for the RAG question-answering endpoint.
All business logic is delegated to the RAGOrchestrator — this file
handles only HTTP concerns: request validation, dependency wiring,
and response serialization.

Endpoint:
  POST /api/v1/chat  →  RAGOrchestrator.process_question()
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.logging import logger
from backend.app.database.session import get_db
from backend.app.dependencies.auth import get_current_user
from backend.app.models.user import User
from backend.app.repositories.chunk_repository import ChunkRepository
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.llm import get_llm_provider
from backend.app.services.prompt_service import PromptService
from backend.app.services.rag_orchestrator import RAGOrchestrator
from backend.app.services.retriever_service import RetrieverService

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Ask a question and receive a grounded answer from indexed documents.

    The endpoint executes the full RAG pipeline:
      1. Embeds the question using the same model as document indexing
      2. Searches pgvector for the most relevant chunks
      3. Constructs a grounded prompt with retrieved context
      4. Calls the configured LLM provider (Gemini by default)
      5. Returns the answer with source citations

    Authentication: JWT Bearer token required.

    Request Body:
        question: The natural-language question to answer.
        document_ids: Optional list to restrict search to specific documents.
        conversation_id: Optional ID for multi-turn sessions (Phase 2).
        top_k: Override for number of chunks to retrieve.
        score_threshold: Override for minimum similarity score.

    Returns:
        ChatResponse with answer, source citations, model info, and retrieval metadata.
    """
    logger.info(
        f"Chat request from user={current_user.id} | "
        f"question_length={len(request.question)}"
    )

    # Wire dependencies — follows the same pattern as documents.py
    chunk_repo = ChunkRepository(db)
    doc_repo = DocumentRepository(db)
    embedding_service = EmbeddingService()
    retriever_service = RetrieverService(chunk_repo=chunk_repo, doc_repo=doc_repo)
    prompt_service = PromptService()
    llm_provider = get_llm_provider()

    orchestrator = RAGOrchestrator(
        embedding_service=embedding_service,
        retriever_service=retriever_service,
        prompt_service=prompt_service,
        llm_provider=llm_provider,
    )

    response = await orchestrator.process_question(request, user_id=current_user.id)

    logger.info(
        f"Chat response generated | user_id={current_user.id} | sources={len(response.sources)} | "
        f"model={response.model}"
    )

    return response

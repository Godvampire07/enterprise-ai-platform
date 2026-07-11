import io
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Query
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.dependencies.auth import get_current_user
from backend.app.models.user import User
from backend.app.schemas.document import DocumentResponse, ChunkResponse, SearchRequest, SearchResponse, SearchResult
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.chunk_repository import ChunkRepository

from backend.app.services.document_ingestion_service import DocumentIngestionService
from backend.app.services.document_service import DocumentService
from backend.app.services.parser_service import ParserService
from backend.app.services.chunking_service import ChunkingService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_service import VectorService

router = APIRouter()

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a PDF document. Paraphrases, chunks, creates embeddings,
    and bulk-inserts segments to Postgres db vector tables.
    """
    if not file.filename.lower().endswith('.pdf') and file.content_type != 'application/pdf':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported at this stage."
        )

    try:
        # Read file bytes to find size and reset stream index
        contents = await file.read()
        file_size = len(contents)
        file_io = io.BytesIO(contents)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded file contents."
        )

    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)
    parser_service = ParserService()
    chunking_service = ChunkingService()
    embedding_service = EmbeddingService()

    ingestion_service = DocumentIngestionService(
        db=db,
        doc_repo=doc_repo,
        chunk_repo=chunk_repo,
        parser_service=parser_service,
        chunking_service=chunking_service,
        embedding_service=embedding_service
    )

    try:
        doc = ingestion_service.ingest_document(
            file_content=file_io,
            original_filename=file.filename,
            content_type=file.content_type or "application/pdf",
            file_size=file_size
        )
        return doc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion pipeline failed: {str(e)}"
        )

@router.get("/{id}", response_model=DocumentResponse)
def get_document_metadata(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve metadata of a database stored document."""
    doc_repo = DocumentRepository(db)
    doc_service = DocumentService(db, doc_repo)
    doc = doc_service.get_document(id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
    return doc

@router.get("/{id}/chunks", response_model=List[ChunkResponse])
def get_chunks(
    id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve text chunks of a specific document (paginated)."""
    doc_repo = DocumentRepository(db)
    doc = doc_repo.get_document(id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
        
    chunk_repo = ChunkRepository(db)
    return chunk_repo.get_chunks(document_id=id, skip=skip, limit=limit)

@router.delete("/{id}", response_model=DocumentResponse)
def delete_document(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a document record, its embedded chunks, and the file on disk."""
    doc_repo = DocumentRepository(db)
    doc_service = DocumentService(db, doc_repo)
    doc = doc_service.delete_document(id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
    return doc

@router.post("/search", response_model=SearchResponse)
def search_chunks(
    request: SearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Search for semantically similar document chunks in the vector database."""
    embedding_service = EmbeddingService()
    chunk_repo = ChunkRepository(db)
    vector_service = VectorService(chunk_repo)

    try:
        # Encode semantic query to vector floats
        query_embedding = embedding_service.embed_text(request.query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate query embedding: {str(e)}"
        )

    # Cosine Similarity Search
    results = vector_service.search_similar_chunks(
        query_embedding=query_embedding,
        document_id=request.document_id,
        top_k=request.top_k,
        min_score=request.min_score
    )

    search_results = []
    for chunk, score in results:
        search_results.append(
            SearchResult(
                chunk=ChunkResponse.model_validate(chunk),
                similarity_score=score,
                source_metadata=chunk.chunk_metadata
            )
        )
    return SearchResponse(results=search_results)

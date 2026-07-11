from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

class DocumentCreate(BaseModel):
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    content_type: str

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    content_type: str
    created_at: datetime

class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    chunk_index: int
    page_number: Optional[int]
    text: str
    chunk_metadata: Dict[str, Any]  
    created_at: datetime

class SearchRequest(BaseModel):
    query: str
    document_id: Optional[int] = None
    top_k: Optional[int] = Field(None, ge=1)
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class SearchResult(BaseModel):
    chunk: ChunkResponse
    similarity_score: float
    source_metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]

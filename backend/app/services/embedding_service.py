import threading
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from backend.app.core.config import settings

class EmbeddingService:
    _model = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.model_name = settings.EMBEDDING_MODEL
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        if EmbeddingService._model is None:
            with EmbeddingService._lock:
                if EmbeddingService._model is None:
                    # Check if CUDA is available, otherwise default to CPU
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    # Load model only once
                    EmbeddingService._model = SentenceTransformer(
                        self.model_name,
                        device=device
                    )

    def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for a single text string."""
        if not text:
            return []
        embedding = EmbeddingService._model.encode(text)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate list of vector embeddings for a list of text strings."""
        if not texts:
            return []
        embeddings = EmbeddingService._model.encode(texts)
        return embeddings.tolist()

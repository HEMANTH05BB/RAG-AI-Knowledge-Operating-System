import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Centralized service for managing and generating text embeddings.
    Uses a singleton-like lazy loaded model instance to minimize memory footprint
    and avoid loading heavy model weights during imports or test suites.
    """
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME

    def _get_model(self) -> SentenceTransformer:
        if EmbeddingService._model is None:
            try:
                EmbeddingService._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer model {self.model_name}: {e}")
                raise RuntimeError(f"Embedding model initialization failed: {e}")
        return EmbeddingService._model

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates embedding vector for a single text string.
        """
        if not text:
            return []
        model = self._get_model()
        try:
            return model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a list of text strings in batch.
        """
        if not texts:
            return []
        model = self._get_model()
        try:
            embeddings = model.encode(texts)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to encode batch of texts: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}")

    def get_dimension(self) -> int:
        """
        Returns the embedding vector dimensionality.
        """
        model = self._get_model()
        return model.get_embedding_dimension()

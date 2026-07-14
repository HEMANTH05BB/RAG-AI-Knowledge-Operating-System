import logging
import uuid
from typing import List, Dict, Any, Optional

from app.services.vector_store import VectorStoreService
from app.retrieval.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class Indexer:
    """
    Centralized service for preparation, batch embedding generation, and vector indexing
    of raw texts/chunks into local Qdrant collections.
    """
    def __init__(
        self, 
        vector_store: Optional[VectorStoreService] = None, 
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.vector_store = vector_store or VectorStoreService()
        self.embedding_service = embedding_service or EmbeddingService()

    def index_segments(
        self, 
        segments: List[Dict[str, Any]], 
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Indexes a list of text segments with associated metadata.
        Each segment should be structured as:
        {
            "text": "content chunk...",
            "metadata": {"source": "filename", ...}
        }
        """
        if not segments:
            return {"status": "skipped", "message": "No text segments provided for indexing."}

        try:
            # 1. Fetch embedding model dimensions and ensure collection exists
            vector_size = self.embedding_service.get_dimension()
            self.vector_store.create_collection(collection_name, vector_size=vector_size)

            # 2. Extract texts and payloads for batch encoding
            texts = [seg["text"] for seg in segments]
            
            # 3. Generate embeddings in a single batch (significantly faster on CPU/GPU)
            embeddings = self.embedding_service.get_embeddings(texts)

            # 4. Compile vector store points
            points = []
            for idx, (text, seg, vector) in enumerate(zip(texts, segments, embeddings)):
                point_id = str(uuid.uuid4())
                meta = seg.get("metadata", {})
                
                # Combine segment metadata with textual payload
                payload = {
                    **meta,
                    "text": text,
                    "chunk_index": idx,
                    "length": len(text)
                }

                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": payload
                })

            # 5. Upsert to Qdrant
            success = self.vector_store.upsert_vectors(collection_name, points)
            if success:
                return {
                    "status": "success",
                    "collection": collection_name,
                    "total_indexed": len(points)
                }
            else:
                raise RuntimeError("Vector database rejected bulk upsert operation.")

        except Exception as e:
            logger.error(f"Error during indexing segments: {e}")
            return {
                "status": "error",
                "message": f"Indexing failure: {str(e)}"
            }

class QdrantIndexer:
    def __init__(self):
        from app.services.vector_store import VectorStoreService
        self.vector_store = VectorStoreService()

    def search(self, query_embedding: list, limit: int = 5, collection_name: str = "knowledge_base"):
        results = self.vector_store.search_vectors(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        class SearchResult:
            def __init__(self, score, payload):
                self.score = score
                self.payload = payload
        return [SearchResult(r["score"], r["payload"]) for r in results]

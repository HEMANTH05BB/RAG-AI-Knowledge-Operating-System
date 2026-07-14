import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.vector_store import VectorStoreService
from app.retrieval.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["Semantic Search"])

class SearchRequest(BaseModel):
    query: str
    collection_name: str = "knowledge_base"
    limit: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None

class SearchResultItem(BaseModel):
    id: str
    score: float
    text: str
    source: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    collection: str
    results: List[SearchResultItem]

@router.post("", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Semantic Search endpoint: performs similarity lookup in Qdrant using the query embedding.
    Returns matched text segments, source filenames, and similarity scores.
    """
    # 1. Generate query embedding
    try:
        embedding_service = EmbeddingService()
        query_vector = embedding_service.get_embedding(request.query)
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode query string: {str(e)}"
        )

    # 2. Query Qdrant vector database
    vector_store = None
    try:
        vector_store = VectorStoreService()
        search_results = vector_store.search_vectors(
            collection_name=request.collection_name,
            query_vector=query_vector,
            limit=request.limit,
            filter_dict=request.filter_metadata
        )
    except Exception as e:
        logger.error(f"Database semantic lookup failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}"
        )
    finally:
        if vector_store and hasattr(vector_store, "client"):
            try:
                vector_store.client.close()
            except Exception:
                pass

    # 3. Format return list
    compiled_results = []
    for res in search_results:
        payload = res.get("payload", {})
        text = payload.get("text", "")
        source = payload.get("source", "unknown")
        
        # Strip text and source from the raw payload dictionary so we return clean metadata
        clean_metadata = {k: v for k, v in payload.items() if k not in ["text", "source"]}
        
        compiled_results.append(SearchResultItem(
            id=str(res.get("id", "")),
            score=float(res.get("score", 0.0)),
            text=text,
            source=source,
            metadata=clean_metadata
        ))

    return SearchResponse(
        query=request.query,
        collection=request.collection_name,
        results=compiled_results
    )

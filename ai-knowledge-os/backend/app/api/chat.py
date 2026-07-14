import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.retrieval.embedding_service import EmbeddingService

from app.config import settings
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str
    collection_name: str = "knowledge_base"
    limit: int = 4

class ChatSource(BaseModel):
    source: str
    text: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]

@router.post("", response_model=ChatResponse)
async def chat_interaction(request: ChatRequest):
    """
    RAG Chat endpoint: retrieves context documents from Qdrant based on the query,
    formats a prompt, and queries the LLM for a contextual answer.
    """
    # 1. Generate query embedding
    try:
        embedding_service = EmbeddingService()
        query_vector = embedding_service.get_embedding(request.message)
    except Exception as e:
        logger.error(f"Failed to load embedding service or generate query vector: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode query message: {str(e)}"
        )
        
    # 2. Retrieve context from Qdrant
    vector_store = None
    search_results = []
    try:
        vector_store = VectorStoreService()
        search_results = vector_store.search_vectors(
            collection_name=request.collection_name,
            query_vector=query_vector,
            limit=request.limit
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        # We don't fail the request immediately; we can fallback to search_results = []
        search_results = []
    finally:
        if vector_store and hasattr(vector_store, "client"):
            try:
                vector_store.client.close()
            except Exception:
                pass
                
    # 3. Compile context strings and metadata sources
    contexts = []
    sources = []
    for res in search_results:
        payload = res.get("payload", {})
        text = payload.get("text", "")
        source = payload.get("source", "unknown")
        score = res.get("score", 0.0)
        
        if text:
            contexts.append(text)
            sources.append(ChatSource(
                source=source,
                text=text,
                score=score
            ))
            
    # 4. Generate response using LLM
    try:
        llm = LLMService()
        # If no contexts are found, LLMService handles it by stating it doesn't know
        answer = llm.generate_rag_response(
            query=request.message,
            contexts=contexts
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {str(e)}"
        )
        
    return ChatResponse(
        answer=answer,
        sources=sources
    )

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.vector_store import VectorStoreService
from app.retrieval.embedding_service import EmbeddingService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["Retrieval Augmented Generation"])

class RAGRequest(BaseModel):
    query: str
    collection_name: str = "knowledge_base"
    limit: int = 4
    min_score: float = 0.0
    temperature: float = 0.3
    system_instruction: Optional[str] = None

class RAGSource(BaseModel):
    id: str
    score: float
    text: str
    source: str
    metadata: Dict[str, Any]

class RAGResponse(BaseModel):
    answer: str
    sources: List[RAGSource]

@router.post("", response_model=RAGResponse)
async def generate_rag_content(request: RAGRequest):
    """
    RAG endpoint: performs semantic search against Qdrant, filters results based on
    a score threshold, constructs a prompt, and queries the LLM for a structured answer.
    """
    # 1. Generate query embedding
    try:
        embedding_service = EmbeddingService()
        query_vector = embedding_service.get_embedding(request.query)
    except Exception as e:
        logger.error(f"Failed to generate query embedding in RAG router: {e}")
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
            limit=request.limit
        )
    except Exception as e:
        logger.error(f"RAG semantic lookup failed: {e}")
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

    # 3. Format context segments and filter based on min_score threshold
    contexts = []
    sources = []
    for res in search_results:
        score = float(res.get("score", 0.0))
        if score < request.min_score:
            continue
            
        payload = res.get("payload", {})
        text = payload.get("text", "")
        source = payload.get("source", "unknown")
        
        # Strip text and source from the raw payload dictionary for cleaner metadata
        clean_metadata = {k: v for k, v in payload.items() if k not in ["text", "source"]}
        
        if text:
            contexts.append(text)
            sources.append(RAGSource(
                id=str(res.get("id", "")),
                score=score,
                text=text,
                source=source,
                metadata=clean_metadata
            ))

    # 4. Generate RAG Answer using LLM
    try:
        llm = LLMService()
        
        # Build prompt with contexts
        formatted_context = "\n\n".join([f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(contexts)])
        prompt = f"""Use the following context snippets to answer the user query. If the context does not contain enough information to answer, state that you do not know based on the context.

---
CONTEXT:
{formatted_context}
---

USER QUERY:
{request.query}

ANSWER:"""

        default_system = "You are a helpful knowledge assistant. Always answer accurately and truthfully based on the provided context."
        sys_inst = request.system_instruction or default_system
        
        answer = llm.generate_response(
            prompt=prompt,
            system_instruction=sys_inst,
            temperature=request.temperature
        )
    except Exception as e:
        logger.error(f"RAG LLM generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {str(e)}"
        )

    return RAGResponse(
        answer=answer,
        sources=sources
    )

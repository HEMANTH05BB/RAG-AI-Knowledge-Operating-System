import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService

router = APIRouter(prefix="/api/summarize", tags=["Summarization"])

class SummarizeRequest(BaseModel):
    filename: str
    collection_name: str = "knowledge_base"

class SummarizeResponse(BaseModel):
    filename: str
    status: str
    summary: str

@router.post("", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest):
    """
    Retrieves all text chunks for a document from Qdrant, compiles them into a
    single context, and uses Llama 3.3 70B via OpenRouter to generate a structured summary.
    """
    try:
        # 1. Retrieve all chunks for this document
        vs_service = VectorStoreService()
        points = vs_service.get_document_chunks(
            collection_name=request.collection_name,
            filename=request.filename
        )
        
        # Explicitly close vector store to avoid SQLite file locks
        vs_service.client.close()
        
        if not points:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{request.filename}' not found or has no chunks in collection '{request.collection_name}'."
            )
            
        # 2. Reconstruct raw document context
        chunks_text = []
        for p in points:
            text_payload = p["payload"].get("text", "")
            if text_payload:
                chunks_text.append(text_payload)
                
        document_context = "\n\n".join(chunks_text)
        
        if not document_context.strip():
            raise HTTPException(
                status_code=404,
                detail=f"No readable text content found in chunks for document '{request.filename}'."
            )
            
        # 3. Compile prompt requesting a structured summary
        prompt = f"""You are an advanced reading and summarization assistant. 
Please analyze the following document context fully and provide a comprehensive structured summary.

Structure your response with:
1. **Title / Topic Overview**: A brief title/topic header.
2. **Key Takeaways**: A bulleted list of the main points, insights, or findings (3-5 points).
3. **Executive Summary**: A concise paragraph summarizing the entire context.

---
DOCUMENT CONTEXT:
{document_context}
---

Your response MUST be clear, objective, and solely based on the document context provided above."""

        # 4. Invoke LLM Service targeting Llama 3.3 70B instruct
        # We pass model_name="meta-llama/llama-3.3-70b-instruct:free" as the primary target
        llm = LLMService(model_name="meta-llama/llama-3.3-70b-instruct:free")
        summary_text = llm.generate(prompt)
        
        return SummarizeResponse(
            filename=request.filename,
            status="success",
            summary=summary_text
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to summarize document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate summary: {str(e)}"
        )

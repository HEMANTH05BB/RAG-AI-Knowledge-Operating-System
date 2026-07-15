import logging
import json
import re
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService

router = APIRouter(prefix="/api/flashcards", tags=["Flashcards"])

class FlashcardItem(BaseModel):
    question: str
    answer: str

class FlashcardRequest(BaseModel):
    filename: str
    limit: int = 5
    collection_name: str = "knowledge_base"

class FlashcardResponse(BaseModel):
    filename: str
    status: str
    flashcards: List[FlashcardItem]

def clean_json_response(text: str) -> str:
    """
    Cleans raw markdown blocks and whitespace from LLM output to extract a parsable JSON array.
    """
    text = text.strip()
    # Strip markdown wrapper if present
    if "```" in text:
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx+1]
    return text

@router.post("", response_model=FlashcardResponse)
async def generate_flashcards(request: FlashcardRequest):
    """
    Retrieves all text chunks for a document from Qdrant, compiles them into a
    single context, and uses Google Gemma via OpenRouter to generate structured study flashcards.
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
            
        # 3. Compile prompt requesting a structured list of flashcards
        prompt = f"""You are an advanced educational study assistant. 
Please analyze the following document context fully and generate exactly {request.limit} high-quality study flashcards (Question & Answer pairs) based on it.

Each flashcard should target a core concept, key term, or main insight from the document.
The response MUST be returned as a valid, parsable JSON array of objects, where each object has exactly two keys: "question" and "answer". Do not include any introductory or concluding text, explanations, or wrappers.

Format example:
[
  {{"question": "What is the primary role of FastAPI?", "answer": "FastAPI is a Python web framework designed to build APIs efficiently."}},
  {{"question": "What database is used?", "answer": "A serverless Qdrant vector database."}}
]

---
DOCUMENT CONTEXT:
{document_context}
---"""

        # 4. Invoke LLM Service targeting Gemma 4 31B
        llm = LLMService(model_name="google/gemma-4-31b-it:free")
        raw_output = llm.generate(prompt)
        
        # 5. Clean and parse JSON response
        cleaned_output = clean_json_response(raw_output)
        try:
            flashcards_data = json.loads(cleaned_output)
            if not isinstance(flashcards_data, list):
                raise ValueError("LLM returned JSON that is not a list of objects")
                
            flashcards = []
            for item in flashcards_data:
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if q and a:
                    flashcards.append(FlashcardItem(question=q, answer=a))
                    
            # Ensure we return at most the requested limit
            flashcards = flashcards[:request.limit]
            
        except Exception as json_err:
            logging.getLogger(__name__).error(f"Failed to parse JSON from LLM output: {json_err}. Raw output was: {raw_output}")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to parse flashcards output from LLM: {str(json_err)}"
            )
            
        return FlashcardResponse(
            filename=request.filename,
            status="success",
            flashcards=flashcards
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to generate flashcards: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate flashcards: {str(e)}"
        )

import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.ingestion.document_ingestor import DocumentIngestor
from app.services.vector_store import VectorStoreService

from app.config import settings

router = APIRouter(prefix="/api/upload", tags=["Upload"])

# Initialize UPLOAD_DIR from settings
UPLOAD_DIR = settings.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form("knowledge_base"),
    chunk_size: int = Form(settings.CHUNK_SIZE),
    chunk_overlap: int = Form(settings.CHUNK_OVERLAP)
):
    """
    Upload a document, extract text, split it into chunks, generate embeddings,
    and index them in the local Qdrant vector database.
    """
    # 1. Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = [".pdf", ".docx", ".pptx", ".html", ".htm", ".txt"]
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(allowed_exts)}"
        )
        
    # 2. Save uploaded file to temporary path
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write file to disk: {str(e)}"
        )
        
    # 3. Run Ingestion Pipeline
    vector_store = None
    try:
        vector_store = VectorStoreService()
        ingestor = DocumentIngestor(vector_store=vector_store)
        
        result = ingestor.ingest_file(
            temp_file_path,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if result["status"] == "success":
            return {
                "message": f"Successfully ingested {file.filename}",
                "filename": file.filename,
                "collection": collection_name,
                "chunks_count": result["total_chunks"]
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Ingestion failed")
            )
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
        
    finally:
        # 4. Clean up connections
        if vector_store and hasattr(vector_store, "client"):
            try:
                vector_store.client.close()
            except Exception:
                pass
        # 5. Clean up temporary uploaded file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

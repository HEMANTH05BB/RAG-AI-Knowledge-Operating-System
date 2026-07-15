import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.services.processing_pipeline import ProcessingPipeline
from app.services.vector_store import VectorStoreService
from app.config import settings

router = APIRouter(prefix="/api/upload", tags=["Upload"])

# Initialize UPLOAD_DIR from settings
UPLOAD_DIR = settings.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/list")
async def list_ingested_documents(collection_name: str = "knowledge_base"):
    """
    Retrieve list of unique document sources currently indexed in the specified Qdrant collection.
    """
    vs = None
    try:
        vs = VectorStoreService()
        # Scroll points to collect payloads (limit 1000)
        results, _ = vs.client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        # Extract unique sources
        sources = set()
        for pt in results:
            source = pt.payload.get("source")
            if source:
                sources.add(source)
                
        return {
            "status": "success",
            "documents": sorted(list(sources))
        }
    except Exception as e:
        # If collection doesn't exist, return empty list gracefully
        return {
            "status": "success",
            "documents": []
        }
    finally:
        if vs and hasattr(vs, "client"):
            try:
                vs.client.close()
            except Exception:
                pass

@router.post("")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form("knowledge_base"),
    chunk_size: int = Form(settings.CHUNK_SIZE),
    chunk_overlap: int = Form(settings.CHUNK_OVERLAP),
    generate_summary: bool = Form(False)
):
    """
    Upload a document, extract text, split it recursively into chunks,
    optionally generate a document summary, and index them in the Qdrant database.
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
    pipeline = None
    try:
        pipeline = ProcessingPipeline()
        result = pipeline.process_file(
            temp_file_path,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            generate_summary=generate_summary
        )
        
        if result["status"] == "success":
            return {
                "message": f"Successfully ingested {file.filename}",
                "filename": file.filename,
                "collection": collection_name,
                "chunks_count": result["total_chunks"],
                "summary": result.get("summary")
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
        # Clean up database client connection
        if pipeline:
            try:
                pipeline.close()
            except Exception:
                pass
        # 4. Clean up temporary uploaded file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

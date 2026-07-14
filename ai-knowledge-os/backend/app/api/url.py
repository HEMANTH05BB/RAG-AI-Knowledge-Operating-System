import logging
import re
import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/url", tags=["URL Ingestion"])


class URLIngestRequest(BaseModel):
    url: str
    collection_name: str = "knowledge_base"
    chunk_size: int = settings.CHUNK_SIZE
    chunk_overlap: int = settings.CHUNK_OVERLAP
    generate_summary: bool = False

class URLIngestResponse(BaseModel):
    status: str
    url: str
    collection: str
    total_chunks: int
    source_type: str
    summary: Optional[str] = None

def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extracts 11-character video ID from a YouTube URL.
    """
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_youtube_transcript(video_id: str) -> str:
    """
    Retrieves captions for a YouTube video using YouTube Transcript API.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript_list])
    except Exception as e:
        logger.error(f"Failed to fetch YouTube transcript for {video_id}: {e}")
        raise ValueError(f"Could not retrieve YouTube transcript. It might be disabled or restricted: {str(e)}")

def fetch_webpage_content(url: str) -> str:
    """
    Scrapes text content from a general HTML page, stripping styles and headers.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        raise ValueError(f"Failed to download webpage: {str(e)}")
        
    soup = BeautifulSoup(response.content, "html.parser")
    # Decompose script, style, header, footer, nav
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
        
    text = soup.get_text()
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = "\n".join(chunk for chunk in chunks if chunk)
    
    if not clean_text.strip():
        raise ValueError("Webpage returned no readable text content.")
        
    return clean_text

@router.post("", response_model=URLIngestResponse)
async def ingest_url_content(request: URLIngestRequest):
    """
    Ingests web URLs or YouTube URLs, extracts their contents, chunks them recursively,
    optionally generates an AI summary, and stores them in Qdrant.
    """
    try:
        from app.services.processing_pipeline import ProcessingPipeline
        pipeline = ProcessingPipeline()
        result = pipeline.process_url(
            url=request.url,
            collection_name=request.collection_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            generate_summary=request.generate_summary
        )
        
        if result["status"] == "success":
            return URLIngestResponse(
                status="success",
                url=request.url,
                collection=request.collection_name,
                total_chunks=result["total_chunks"],
                source_type=result["source_type"],
                summary=result.get("summary")
            )
        else:
            err_msg = result.get("message", "Ingestion failed")
            status_code = 400
            if "Vector storage" in err_msg or "Indexing" in err_msg or "Qdrant" in err_msg:
                status_code = 500
            raise HTTPException(
                status_code=status_code,
                detail=err_msg
            )
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index URL contents: {str(e)}"
        )

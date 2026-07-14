import logging
import re
import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.indexer import Indexer

from app.config import settings
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/url", tags=["URL Ingestion"])


class URLIngestRequest(BaseModel):
    url: str
    collection_name: str = "knowledge_base"
    chunk_size: int = settings.CHUNK_SIZE
    chunk_overlap: int = settings.CHUNK_OVERLAP

class URLIngestResponse(BaseModel):
    status: str
    url: str
    collection: str
    total_chunks: int
    source_type: str

def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extracts 11-character video ID from a YouTube URL.
    """
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    # Try custom Shorts match
    shorts_match = re.search(r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})', url)
    if shorts_match:
        return shorts_match.group(1)
    return None

def fetch_youtube_transcript(video_id: str) -> str:
    """
    Fetches transcript text for a given YouTube video ID.
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

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits content string into recursive overlapping chunks.
    """
    chunker = TextChunker()
    return chunker.split_recursively(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

@router.post("", response_model=URLIngestResponse)
async def ingest_url_content(request: URLIngestRequest):
    """
    Ingests web URLs or YouTube URLs, extracts their contents, chunks them,
    generates vector embeddings, and stores them in Qdrant.
    """
    try:
        embedding_service = EmbeddingService()
    except Exception as e:
        logger.error(f"Failed to load embedding service: {e}")
        raise HTTPException(
            status_code=500,
            detail="Embedding service is not initialized."
        )

    # 1. Fetch content depending on URL pattern
    url_str = request.url
    youtube_id = extract_youtube_video_id(url_str)
    
    try:
        if youtube_id:
            raw_text = fetch_youtube_transcript(youtube_id)
            source_type = "youtube_transcript"
            source_name = f"YouTube (ID: {youtube_id})"
        else:
            raw_text = fetch_webpage_content(url_str)
            source_type = "webpage"
            source_name = url_str
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # 2. Chunk text content
    chunks = split_text_into_chunks(
        raw_text, 
        chunk_size=request.chunk_size, 
        chunk_overlap=request.chunk_overlap
    )
    
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No contents could be chunked or extracted from the URL."
        )

    # 3. Format segments for indexer
    segments = []
    for chunk in chunks:
        segments.append({
            "text": chunk,
            "metadata": {
                "source": source_name,
                "url": url_str,
                "source_type": source_type
            }
        })

    # 4. Batch index via Indexer
    try:
        indexer = Indexer(embedding_service=embedding_service)
        result = indexer.index_segments(segments, request.collection_name)
        if result["status"] != "success":
            raise RuntimeError(result.get("message", "Vector store rejected upsert operation."))
    except Exception as e:
        logger.error(f"Failed to ingest URL: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index URL contents: {str(e)}"
        )

    return URLIngestResponse(
        status="success",
        url=url_str,
        collection=request.collection_name,
        total_chunks=len(chunks),
        source_type=source_type
    )

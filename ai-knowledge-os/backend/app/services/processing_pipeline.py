import logging
import os
from typing import List, Dict, Any, Optional

from app.ingestion.document_ingestor import DocumentIngestor
from app.retrieval.indexer import Indexer
from app.retrieval.chunker import TextChunker
from app.services.llm_service import LLMService
from app.api.url import fetch_webpage_content, fetch_youtube_transcript, extract_youtube_video_id

logger = logging.getLogger(__name__)

class ProcessingPipeline:
    """
    Unified coordinator service that orchestrates different ingestion methods (files, URLs, raw text),
    performs text chunking and vector indexing, and optionally enriches metadata using LLM summaries.
    """
    def __init__(
        self, 
        indexer: Optional[Indexer] = None, 
        chunker: Optional[TextChunker] = None, 
        llm: Optional[LLMService] = None
    ):
        self.indexer = indexer or Indexer()
        self.chunker = chunker or TextChunker()
        self.llm = llm or LLMService()
        self.document_ingestor = DocumentIngestor(
            vector_store=self.indexer.vector_store,
            model_name=self.indexer.embedding_service.model_name
        )

    def process_file(
        self, 
        file_path: str, 
        collection_name: str, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100,
        generate_summary: bool = False
    ) -> Dict[str, Any]:
        """
        Ingests a document file, extracts its contents, splits recursively,
        optionally appends an LLM-generated summary, and indexes in Qdrant.
        """
        filename = os.path.basename(file_path)
        try:
            # 1. Extract raw sections
            sections = self.document_ingestor.extract_text(file_path)
            if not sections:
                return {"status": "skipped", "message": "No text extracted from file", "filename": filename}
            
            # Combine all text for summary generation if requested
            full_text = "\n".join([sec["text"] for sec in sections])
            doc_summary = self._generate_document_summary(full_text) if generate_summary else None

            # 2. Chunk text and prepare segments
            segments = []
            for sec in sections:
                sec_text = sec["text"]
                sec_metadata = sec["metadata"]
                if doc_summary:
                    sec_metadata["document_summary"] = doc_summary
                
                chunks = self.chunker.split_recursively(sec_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                for chunk in chunks:
                    segments.append({
                        "text": chunk,
                        "metadata": sec_metadata
                    })

            # 3. Index segments via indexer
            result = self.indexer.index_segments(segments, collection_name)
            if result["status"] == "success":
                return {
                    "status": "success",
                    "filename": filename,
                    "collection": collection_name,
                    "total_chunks": result["total_indexed"],
                    "summary": doc_summary
                }
            return {"status": "error", "message": result.get("message", "Vector storage failed"), "filename": filename}
        except Exception as e:
            logger.error(f"Failed to process file in pipeline: {e}")
            return {"status": "error", "message": str(e), "filename": filename}

    def process_url(
        self, 
        url: str, 
        collection_name: str, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100,
        generate_summary: bool = False
    ) -> Dict[str, Any]:
        """
        Ingests web URLs or YouTube transcripts, splits recursively,
        optionally appends an LLM-generated summary, and indexes in Qdrant.
        """
        try:
            # 1. Extract content depending on URL pattern
            youtube_id = extract_youtube_video_id(url)
            if youtube_id:
                raw_text = fetch_youtube_transcript(youtube_id)
                source_type = "youtube_transcript"
                source_name = f"YouTube (ID: {youtube_id})"
            else:
                raw_text = fetch_webpage_content(url)
                source_type = "webpage"
                source_name = url
                
            doc_summary = self._generate_document_summary(raw_text) if generate_summary else None

            # 2. Chunk content
            chunks = self.chunker.split_recursively(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # 3. Format segments
            segments = []
            for chunk in chunks:
                metadata = {
                    "source": source_name,
                    "url": url,
                    "source_type": source_type
                }
                if doc_summary:
                    metadata["document_summary"] = doc_summary

                segments.append({
                    "text": chunk,
                    "metadata": metadata
                })

            # 4. Index segments via indexer
            result = self.indexer.index_segments(segments, collection_name)
            if result["status"] == "success":
                return {
                    "status": "success",
                    "url": url,
                    "collection": collection_name,
                    "total_chunks": result["total_indexed"],
                    "source_type": source_type,
                    "summary": doc_summary
                }
            return {"status": "error", "message": result.get("message", "Vector storage failed"), "url": url}
        except Exception as e:
            logger.error(f"Failed to process URL in pipeline: {e}")
            return {"status": "error", "message": str(e), "url": url}

    def _generate_document_summary(self, text: str) -> Optional[str]:
        """
        Generates a 2-3 sentence summary of the document using LLM.
        Takes the first 4000 characters to stay within context limits.
        """
        if not text:
            return None
        truncated_text = text[:4000]
        prompt = f"Please summarize the following document context in 2-3 sentences. Keep it brief, factual, and direct:\n\n{truncated_text}"
        try:
            return self.llm.generate_response(prompt, temperature=0.3)
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return None

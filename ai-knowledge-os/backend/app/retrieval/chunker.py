import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Centralized utility class to split text into overlapping semantic chunks
    suitable for indexing in a vector database.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_by_character(self, text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
        """
        Simple character-based chunking with fixed overlap.
        """
        size = chunk_size or self.chunk_size
        overlap = chunk_overlap or self.chunk_overlap
        
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            
            if end >= text_len:
                break
                
            start += (size - overlap)
            if start >= text_len or size <= overlap:
                break
                
        return [c for c in chunks if c]

    def split_recursively(
        self, 
        text: str, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ) -> List[str]:
        """
        Splits text recursively using a list of separators (e.g. paragraphs, lines, sentences).
        This attempts to keep semantic blocks (like sentences or paragraphs) in single chunks.
        """
        size = chunk_size or self.chunk_size
        overlap = chunk_overlap or self.chunk_overlap
        seps = separators or ["\n\n", "\n", ". ", " ", ""]
        
        if not text:
            return []
            
        return self._split_text(text, seps, size, overlap)

    def _split_text(self, text: str, separators: List[str], max_size: int, overlap: int) -> List[str]:
        # Base case: text is already small enough
        if len(text) <= max_size:
            return [text]

        # Get first separator
        separator = separators[0]
        next_separators = separators[1:] if len(separators) > 1 else [separator]

        # Split text by separator
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        chunks = []
        current_chunk = []
        current_len = 0

        for split in splits:
            # Re-attach the split separator (except for last element)
            split_with_sep = split + separator if separator != "" and split != splits[-1] else split
            split_len = len(split_with_sep)

            # If a single split item is larger than max_size, recursively split it
            if split_len > max_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Split recursively with the remaining separators
                sub_chunks = self._split_text(split_with_sep, next_separators, max_size, overlap)
                chunks.extend(sub_chunks)
            else:
                # Check if adding this split item exceeds max size
                if current_len + split_len > max_size:
                    chunks.append("".join(current_chunk))
                    
                    # Approximate overlap from the end of the flushed chunk
                    overlap_text = "".join(current_chunk)
                    if len(overlap_text) > overlap:
                        overlap_start = len(overlap_text) - overlap
                        # Align overlap start with word boundary
                        space_idx = overlap_text.find(" ", overlap_start)
                        if space_idx != -1 and space_idx < len(overlap_text) - 5:
                            overlap_text = overlap_text[space_idx + 1:]
                        else:
                            overlap_text = overlap_text[overlap_start:]
                            
                    current_chunk = [overlap_text, split_with_sep]
                    current_len = len(overlap_text) + split_len
                else:
                    current_chunk.append(split_with_sep)
                    current_len += split_len

        if current_chunk:
            chunks.append("".join(current_chunk))

        # Filter empty chunks and strip outer whitespaces
        return [c.strip() for c in chunks if c.strip()]

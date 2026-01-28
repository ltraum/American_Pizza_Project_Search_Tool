"""
Sentence chunking utility for creating overlapping two-sentence windows
Inspired by index.py approach for better semantic search granularity
"""
import re
from typing import List, Dict, Any, Tuple


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation markers.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences (stripped of whitespace)
    """
    if not text or not text.strip():
        return []
    
    # Split on sentence-ending punctuation, keeping the punctuation
    # This regex handles: . ! ? followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Filter out empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def create_overlapping_sentence_pairs(
    sentences: List[str],
    window_size: int = 2,
    overlap: int = 1
) -> List[Dict[str, Any]]:
    """
    Create overlapping windows of sentences for better semantic matching.
    
    This creates sliding windows where each window contains 'window_size' sentences,
    and consecutive windows overlap by 'overlap' sentences.
    
    Example with window_size=2, overlap=1:
        Sentences: [s1, s2, s3, s4, s5]
        Windows:   [s1+s2, s2+s3, s3+s4, s4+s5]
    
    Args:
        sentences: List of sentences
        window_size: Number of sentences per window (default: 2 for two-sentence pairs)
        overlap: Number of sentences to overlap between windows (default: 1)
        
    Returns:
        List of chunk dicts with:
            - text: Combined text of the sentence pair
            - start_sentence_idx: Index of first sentence in pair
            - end_sentence_idx: Index of last sentence in pair
            - sentence_count: Number of sentences in this chunk
    """
    if not sentences:
        return []
    
    # Overlap cannot be >= window_size (would make step <= 0)
    overlap = min(overlap, max(0, window_size - 1))
    
    if len(sentences) < window_size:
        # If we have fewer sentences than window_size, create a single chunk
        return [{
            'text': ' '.join(sentences),
            'start_sentence_idx': 0,
            'end_sentence_idx': len(sentences) - 1,
            'sentence_count': len(sentences)
        }]
    
    chunks = []
    step = window_size - overlap  # How many sentences to advance each step
    
    for i in range(0, len(sentences), step):
        end_idx = min(i + window_size, len(sentences))
        window_sentences = sentences[i:end_idx]
        
        if not window_sentences:
            break
        
        chunk_text = ' '.join(window_sentences)
        
        chunks.append({
            'text': chunk_text,
            'start_sentence_idx': i,
            'end_sentence_idx': end_idx - 1,
            'sentence_count': len(window_sentences)
        })
        
        # If we've reached the end, break
        if end_idx >= len(sentences):
            break
    
    return chunks


def chunk_document_into_sentence_pairs(
    text: str,
    window_size: int = 2,
    overlap: int = 1,
    min_sentence_length: int = 10
) -> List[Dict[str, Any]]:
    """
    Chunk a document into overlapping two-sentence windows.
    
    This is the main function to use for creating sentence-pair chunks from documents.
    
    Args:
        text: Full document text
        window_size: Number of sentences per window (default: 2)
        overlap: Number of sentences to overlap (default: 1)
        min_sentence_length: Minimum character length for a sentence to be included
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if not text or not text.strip():
        return []
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    # Filter out very short sentences (likely noise)
    sentences = [s for s in sentences if len(s) >= min_sentence_length]
    
    if not sentences:
        return []
    
    # Create overlapping windows
    chunks = create_overlapping_sentence_pairs(sentences, window_size, overlap)
    
    return chunks


def create_multi_level_chunks(
    text: str,
    small_window: int = 2,
    large_window: int = 4,
    overlap: int = 1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create chunks at multiple granularities for better search coverage.
    
    This creates both small (2-sentence) and large (4-sentence) chunks,
    allowing searches to match both fine-grained details and broader context.
    
    Args:
        text: Full document text
        small_window: Sentences per small chunk (default: 2)
        large_window: Sentences per large chunk (default: 4)
        overlap: Overlap between chunks (default: 1)
        
    Returns:
        Tuple of (small_chunks, large_chunks)
    """
    sentences = split_into_sentences(text)
    sentences = [s for s in sentences if len(s) >= 10]  # Filter short sentences
    
    if not sentences:
        return [], []
    
    small_chunks = create_overlapping_sentence_pairs(sentences, small_window, overlap)
    large_chunks = create_overlapping_sentence_pairs(sentences, large_window, overlap)
    
    return small_chunks, large_chunks

from typing import List, Optional
from pydantic import BaseModel, Field


class SparseVector(BaseModel):
    """Sparse vector representation for BM25 embeddings."""
    indices: List[int] = Field(..., description="Indices of active vocabulary tokens")
    values: List[float] = Field(..., description="TF-IDF / BM25 weights corresponding to indices")


class QueryEmbedRequest(BaseModel):
    """Request payload for search query embedding."""
    text: str = Field(..., min_length=1, description="Query text to embed for search retrieval")


class QueryEmbedResponse(BaseModel):
    """Response containing ColQwen multi-vector and BM25 sparse vector for a query."""
    colqwen: List[List[float]] = Field(..., description="Multi-vector token embeddings from ColQwen2.5")
    bm25: SparseVector = Field(..., description="Sparse BM25 token representation")
    dense: Optional[List[List[float]]] = Field(default=None, description="Alias for colqwen")
    sparse: Optional[SparseVector] = Field(default=None, description="Alias for bm25")


class ImageEmbedRequest(BaseModel):
    """Request payload for PDF page image batch embedding during ingestion."""
    images_base64: List[str] = Field(..., min_length=1, description="List of base64-encoded page images")
    texts: Optional[List[str]] = Field(default=None, description="Optional extracted page texts for sparse vector indexing")


class ImageEmbedResponse(BaseModel):
    """Response containing multi-vectors for image batch and optional BM25 vectors."""
    colqwen: List[List[List[float]]] = Field(..., description="List of multi-vector embeddings per page image")
    bm25: List[SparseVector] = Field(default_factory=list, description="List of sparse BM25 vectors per page")

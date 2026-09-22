import os
import time
import uuid
import hashlib
import logging
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient, models

logger = logging.getLogger("cache_service")

CACHE_COLLECTION_NAME = "qa_semantic_cache"

def _get_qdrant_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST")
    if not host:
        if os.getenv("KUBERNETES_SERVICE_HOST") or os.getenv("QDRANT_SERVICE_HOST"):
            host = "qdrant"
        else:
            host = "localhost"

    raw_port = os.getenv("QDRANT_PORT", "6333")
    if str(raw_port).startswith("tcp://"):
        raw_port = str(raw_port).split(":")[-1]

    try:
        port = int(raw_port)
    except (ValueError, TypeError):
        port = 6333

    return QdrantClient(host=host, port=port)

qdrant_client = _get_qdrant_client()

def ensure_cache_collection():
    """Ensure the qa_semantic_cache collection exists in Qdrant with proper indexes."""
    try:
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if CACHE_COLLECTION_NAME not in collections:
            logger.info(f"Creating Qdrant Semantic Cache collection '{CACHE_COLLECTION_NAME}'...")
            qdrant_client.create_collection(
                collection_name=CACHE_COLLECTION_NAME,
                vectors_config={
                    "colqwen": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }
            )
            # Create payload indices for fast exact filtering
            qdrant_client.create_payload_index(
                collection_name=CACHE_COLLECTION_NAME,
                field_name="image_hash",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            qdrant_client.create_payload_index(
                collection_name=CACHE_COLLECTION_NAME,
                field_name="query_normalized",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Collection '{CACHE_COLLECTION_NAME}' initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not verify/create Qdrant cache collection: {e}")

def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash for raw image bytes."""
    if not image_bytes:
        return "none"
    return hashlib.sha256(image_bytes).hexdigest()

def normalize_text(text: str) -> str:
    """Normalize text for exact match lookup."""
    return " ".join(text.lower().strip().split())

def check_cache_exact(image_hash: str, query_text: str) -> Optional[Dict[str, Any]]:
    """
    Tier 1: Instant Exact Match Lookup (<1ms).
    Checks if the exact same question was asked for this image without requiring embedding compute.
    """
    start_time = time.perf_counter()
    normalized = normalize_text(query_text)
    try:
        must_filters = [
            models.FieldCondition(key="image_hash", match=models.MatchValue(value=image_hash)),
            models.FieldCondition(key="query_normalized", match=models.MatchValue(value=normalized))
        ]
        res = qdrant_client.scroll(
            collection_name=CACHE_COLLECTION_NAME,
            scroll_filter=models.Filter(must=must_filters),
            limit=1
        )[0]
        if res:
            duration_ms = (time.perf_counter() - start_time) * 1000
            hit_payload = res[0].payload
            logger.info(f"[LATENCY] Qdrant Exact Cache HIT in {duration_ms:.2f}ms (Image Hash: {image_hash[:10]}...)")
            return {
                "answer": hit_payload.get("answer", ""),
                "retrieved_images": hit_payload.get("retrieved_images", []),
                "cached_query": hit_payload.get("query_text", ""),
                "match_type": "exact",
                "score": 1.0
            }
    except Exception as e:
        logger.warning(f"Error checking exact cache: {e}")
    return None

def check_cache_semantic(
    image_hash: str,
    query_text: str,
    query_emb: Dict[str, Any],
    similarity_threshold: float = 0.80
) -> Optional[Dict[str, Any]]:
    """
    Tier 2: Semantic Vector Match Lookup (~10-20ms).
    Checks if a semantically similar question was asked for the same image using ColQwen MaxSim Cosine.
    """
    start_time = time.perf_counter()
    colqwen_vec = query_emb.get("colqwen") or query_emb.get("dense")

    if not colqwen_vec or len(colqwen_vec) == 0:
        return None

    try:
        search_result = qdrant_client.query_points(
            collection_name=CACHE_COLLECTION_NAME,
            query=colqwen_vec,
            using="colqwen",
            query_filter=models.Filter(
                must=[models.FieldCondition(key="image_hash", match=models.MatchValue(value=image_hash))]
            ),
            limit=1
        )

        if search_result.points:
            top_hit = search_result.points[0]
            # ColQwen MaxSim raw score is the sum of maximum cosine similarities across all query tokens.
            # Normalized similarity = score / len(colqwen_vec) (between 0.0 and 1.0)
            normalized_similarity = top_hit.score / len(colqwen_vec)

            if normalized_similarity >= similarity_threshold:
                duration_ms = (time.perf_counter() - start_time) * 1000
                hit_payload = top_hit.payload
                cached_q = hit_payload.get("query_text", "")
                logger.info(
                    f"[LATENCY] Qdrant Semantic Cache HIT (Sim: {normalized_similarity:.4f} >= {similarity_threshold}) "
                    f"in {duration_ms:.2f}ms (Matching '{cached_q}')"
                )
                return {
                    "answer": hit_payload.get("answer", ""),
                    "retrieved_images": hit_payload.get("retrieved_images", []),
                    "cached_query": cached_q,
                    "match_type": "semantic",
                    "score": normalized_similarity
                }
            else:
                logger.info(
                    f"[CACHE] Semantic match candidate below threshold: {normalized_similarity:.4f} < {similarity_threshold} "
                    f"for query '{query_text}' vs cached '{top_hit.payload.get('query_text', '')}'"
                )
    except Exception as e:
        logger.warning(f"Error checking semantic cache: {e}")

    return None

def save_to_cache(
    image_hash: str,
    query_text: str,
    query_emb: Dict[str, Any],
    answer: str,
    retrieved_images: List[str]
):
    """
    Save a question-answer pair and retrieved images to Qdrant Semantic Cache.
    """
    if not answer or not answer.strip():
        return

    colqwen_vec = query_emb.get("colqwen") or query_emb.get("dense", [])
    bm25_vec = query_emb.get("bm25") or query_emb.get("sparse", {"indices": [], "values": []})

    if not colqwen_vec:
        return

    try:
        point_id = str(uuid.uuid4())
        qdrant_client.upsert(
            collection_name=CACHE_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "colqwen": colqwen_vec,
                        "bm25": models.SparseVector(
                            indices=bm25_vec["indices"],
                            values=bm25_vec["values"]
                        )
                    },
                    payload={
                        "image_hash": image_hash,
                        "query_text": query_text,
                        "query_normalized": normalize_text(query_text),
                        "answer": answer,
                        "retrieved_images": retrieved_images,
                        "timestamp": time.time()
                    }
                )
            ]
        )
        logger.info(f"Saved Q&A pair to Qdrant cache: id={point_id}, query='{query_text[:30]}...'")
    except Exception as e:
        logger.error(f"Failed to save to Qdrant cache: {e}")

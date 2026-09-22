import os
import time
import logging
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger("rag_service")

try:
    from backend.core.config import settings
    DEFAULT_EMBEDDER_URL = settings.EMBEDDER_URL
    DEFAULT_EMBEDDER_API_KEY = settings.EMBEDDER_API_KEY
except ImportError:
    DEFAULT_EMBEDDER_URL = "http://localhost:8000"
    DEFAULT_EMBEDDER_API_KEY = ""

EMBEDDER_URL = os.getenv("EMBEDDER_URL", DEFAULT_EMBEDDER_URL)
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY", DEFAULT_EMBEDDER_API_KEY).strip()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "scientific_documents")

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

# Best-practice Score Diffusion parameters from empirical experiments
DIFFUSION_FORWARD = 0.3   # Propagate score to next page (P + 1)
DIFFUSION_BACKWARD = 0.05  # Propagate score to previous page (P - 1)

def encode_query(query_text: str):
    """
    Send text query to Embedder Microservice to convert into Multi-vector and Sparse Vector.
    Supports secure authentication via X-API-Key if EMBEDDER_API_KEY is configured.
    """
    start_time = time.perf_counter()
    try:
        headers = {}
        if EMBEDDER_API_KEY:
            headers["X-API-Key"] = EMBEDDER_API_KEY
            
        response = requests.post(
            f"{EMBEDDER_URL}/embed",
            json={"text": query_text},
            headers=headers,
            timeout=10
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        if response.status_code == 200:
            data = response.json()
            remote_process_ms = response.headers.get("X-Process-Time-Ms", "N/A")
            logger.info(f"[LATENCY] ColQwen Embedder roundtrip: {duration_ms:.2f}ms (Server compute: {remote_process_ms}ms)")
            return {
                "colqwen": data.get("colqwen") or data.get("dense", []),
                "bm25": data.get("bm25") or data.get("sparse", {"indices": [], "values": []})
            }
        else:
            logger.error(f"[LATENCY] Embedder API returned error ({response.status_code}) after {duration_ms:.2f}ms: {response.text}")
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"[LATENCY] Failed to connect to Embedder microservice after {duration_ms:.2f}ms: {e}")
        
    return {"colqwen": [], "bm25": {"indices": [], "values": []}}

def search_qdrant(query_vectors, top_k=3):
    """
    Production Hybrid Search combining ColQwen Multi-vector MaxSim, Sparse BM25, and Neighbor Score Diffusion.
    """
    start_time = time.perf_counter()
    try:
        colqwen_vec = query_vectors.get("colqwen") or query_vectors.get("dense")
        bm25_vec = query_vectors.get("bm25") or query_vectors.get("sparse")

        if not colqwen_vec or not bm25_vec:
            return []

        # 1. Prefetch both branches (ColQwen MaxSim & BM25 Sparse)
        prefetch = [
            models.Prefetch(
                query=colqwen_vec,
                using="colqwen",
                limit=20
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=bm25_vec["indices"],
                    values=bm25_vec["values"]
                ),
                using="bm25",
                limit=20
            )
        ]

        # 2. Qdrant RRF Fusion
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20
        )

        if not search_result.points:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"[LATENCY] Qdrant Search returned 0 points in {duration_ms:.2f}ms")
            return []

        # 3. Neighbor Score Diffusion algorithm
        fused_scores = {}
        payload_cache = {}
        doc_name_default = None

        for rank_idx, hit in enumerate(search_result.points):
            p = hit.payload.get("page_number") or hit.payload.get("page")
            if not p:
                continue
            
            payload_cache[p] = hit.payload
            if not doc_name_default:
                doc_name_default = hit.payload.get("document_name")

            base_score = 1.0 / (rank_idx + 1)
            total_pages = hit.payload.get("total_pages", 999999)

            # Base score for the current page
            fused_scores[p] = fused_scores.get(p, 0.0) + base_score

            # Propagate score to the next page (P + 1)
            if p + 1 <= total_pages:
                fused_scores[p + 1] = fused_scores.get(p + 1, 0.0) + (base_score * DIFFUSION_FORWARD)

            # Propagate score to the previous page (P - 1)
            if p - 1 >= 1:
                fused_scores[p - 1] = fused_scores.get(p - 1, 0.0) + (base_score * DIFFUSION_BACKWARD)

        # Re-rank pages based on diffused total scores
        ranked_pages = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        top_selected_pages = ranked_pages[:top_k]

        # Sort pages in natural ascending order for coherent generator reasoning
        top_selected_pages.sort()

        # 4. Extract image URLs
        image_urls = []
        for p in top_selected_pages:
            if p in payload_cache and payload_cache[p].get("image_url"):
                url = payload_cache[p]["image_url"]
                if url not in image_urls:
                    image_urls.append(url)
            else:
                # Fallback: fetch image_url via scroll if page was promoted via diffusion
                try:
                    must_filters = [
                        models.FieldCondition(key="page_number", match=models.MatchValue(value=p))
                    ]
                    if doc_name_default:
                        must_filters.append(
                            models.FieldCondition(key="document_name", match=models.MatchValue(value=doc_name_default))
                        )
                    res = qdrant_client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=models.Filter(must=must_filters),
                        limit=1
                    )[0]
                    if res and res[0].payload.get("image_url"):
                        url = res[0].payload["image_url"]
                        if url not in image_urls:
                            image_urls.append(url)
                except Exception as e:
                    logger.warning(f"Error fetching neighbor page {p}: {e}")

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"[LATENCY] Qdrant Hybrid Search & Diffusion took {duration_ms:.2f}ms (Retrieved {len(image_urls)} page URLs: {top_selected_pages})")
        return image_urls
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"[LATENCY] Qdrant search error after {duration_ms:.2f}ms: {e}")
        return []

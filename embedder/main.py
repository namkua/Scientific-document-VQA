import os
import io
import time
import base64
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from PIL import Image
import torch

from schemas import (
    QueryEmbedRequest,
    QueryEmbedResponse,
    ImageEmbedRequest,
    ImageEmbedResponse,
    SparseVector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("embedder")

app = FastAPI(
    title="ColQwen Embedder Microservice",
    description="Dedicated microservice for ColQwen2.5 multi-vector and BM25 sparse embeddings.",
    version="1.0.0",
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measure inference and processing latency for every request."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    logger.info(f"[LATENCY] {request.method} {request.url.path} took {duration_ms:.2f}ms (Status: {response.status_code})")
    return response

API_KEY = os.getenv("EMBEDDER_API_KEY", "").strip()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Validate optional API key for client authentication."""
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
    return api_key


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "vidore/colqwen2.5-v0.2"
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0").lower() in ("1", "true", "yes")

model, processor, bm25_model = None, None, None

if not SKIP_MODEL_LOAD:
    print(f"Loading ColQwen2.5 and BM25 on {device}...")
    try:
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        from fastembed import SparseTextEmbedding

        model = ColQwen2_5.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device if device != "mps" else None,
        )
        if device == "mps":
            model = model.to("mps")
        model.eval()
        processor = ColQwen2_5_Processor.from_pretrained(MODEL_ID)
        bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        model, processor, bm25_model = None, None, None


@app.get("/health", tags=["System"])
def health_check():
    """Service health and hardware status."""
    return {
        "status": "ok" if model is not None else "degraded",
        "device": device,
        "auth_enabled": bool(API_KEY),
    }


@app.post("/embed", response_model=QueryEmbedResponse, dependencies=[Depends(verify_api_key)], tags=["Inference"])
async def embed_query(req: QueryEmbedRequest):
    """
    Embed search query for chat / RAG retrieval.
    Generates ColQwen2.5 multi-vector embeddings and BM25 sparse vectors.
    """
    if not model or not processor or not bm25_model:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Generate ColQwen query multi-vector
    with torch.no_grad():
        if hasattr(processor, "process_queries"):
            inputs = processor.process_queries([req.text]).to(device)
        else:
            inputs = processor(text=[req.text], return_tensors="pt").to(device)
        q_emb = model(**inputs)[0].to("cpu").to(torch.float32).numpy().tolist()

    # Generate BM25 sparse vector
    sparse_embedding = list(bm25_model.embed([req.text]))[0]
    sparse_obj = SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )

    return QueryEmbedResponse(
        colqwen=q_emb,
        bm25=sparse_obj,
        dense=q_emb,
        sparse=sparse_obj,
    )


@app.post("/embed-images", response_model=ImageEmbedResponse, dependencies=[Depends(verify_api_key)], tags=["Inference"])
async def embed_images(req: ImageEmbedRequest):
    """
    Embed batch of PDF page images for document ingestion.
    Decodes base64 images and generates ColQwen2.5 multi-vectors and optional BM25 sparse vectors.
    """
    if not model or not processor or not bm25_model:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Decode base64 image strings to PIL images
    pil_images = []
    for b64_str in req.images_base64:
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Generate ColQwen document embeddings
    with torch.no_grad():
        if hasattr(processor, "process_images"):
            batch_inputs = processor.process_images(pil_images).to(device)
        else:
            batch_inputs = processor(images=pil_images, return_tensors="pt").to(device)

        image_embeddings = model(**batch_inputs)
        colqwen_embs = [emb.to("cpu").to(torch.float32).numpy().tolist() for emb in image_embeddings]

    # Generate optional BM25 sparse embeddings if text is provided
    bm25_results = []
    if req.texts:
        for t in req.texts:
            sp = list(bm25_model.embed([t]))[0]
            bm25_results.append(
                SparseVector(
                    indices=sp.indices.tolist(),
                    values=sp.values.tolist(),
                )
            )

    return ImageEmbedResponse(
        colqwen=colqwen_embs,
        bm25=bm25_results,
    )

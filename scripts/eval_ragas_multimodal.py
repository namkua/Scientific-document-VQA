#!/usr/bin/env python3
"""
Scientific Document VQA - Multimodal RAGAS Evaluation Script
Pure Pipeline Evaluation (No Mock/Fallback):
  - Input: user_input (query) from test set
  - Retrieval: ColQwen + Qdrant -> MinIO page images (retrieved_contexts)
  - Generation: Qwen-VL via FastAPI Backend (response)
  - Metrics: MultiModalFaithfulness & MultiModalRelevance via Cloud LLM Judge
"""

import os
import sys
import json
import asyncio
import argparse
import time
import urllib.parse
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from ragas.llms.base import llm_factory

# Import Ragas Multimodal Metrics
try:
    from ragas.metrics.collections import MultiModalFaithfulness, MultiModalRelevance
except ImportError:
    try:
        from ragas.metrics import MultiModalFaithfulness, MultiModalRelevance
    except ImportError:
        print("[!] Error: 'ragas' not installed or outdated. Please install via: pip install 'ragas>=0.2.0'")
        sys.exit(1)


def init_judge_llm(provider: str, model_name: Optional[str] = None):
    """Initializes a vision-capable Cloud LLM Judge for Ragas."""
    provider = provider.lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        client = AsyncOpenAI(api_key=api_key)
        model = model_name or "gpt-4o-mini"
        print(f"[*] Initialized OpenAI Cloud Judge: {model}")
        return llm_factory(model, client=client)

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        model = model_name or "gemini-2.0-flash"
        print(f"[*] Initialized Google Gemini Cloud Judge: {model}")
        return llm_factory(model, client=client)

    elif provider == "litellm":
        base_url = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
        api_key = os.getenv("LITELLM_API_KEY", "sk-litellm-secret-key")
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        model = model_name or "groq/qwen/qwen3.6-27b"
        print(f"[*] Initialized LiteLLM Gateway Judge: {model} at {base_url}")
        return llm_factory(model, client=client)

    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'openai', 'gemini', or 'litellm'")


def download_image_from_minio(url: str, output_dir: str) -> str:
    """
    Downloads image from MinIO to local disk so Cloud Vision API can access it.
    """
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(url):
        return url

    # Method 1: Try MinioService directly from backend
    try:
        from backend.services.storage import storage_service
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lstrip("/")
        parts = path.split("/", 1)
        bucket_name = parts[0] if len(parts) == 2 else None
        object_name = urllib.parse.unquote(parts[1]) if len(parts) == 2 else urllib.parse.unquote(path)
        raw_bytes = storage_service.get_file_bytes(object_name, bucket_name=bucket_name)
        if raw_bytes:
            filename = os.path.basename(object_name) or f"minio_{abs(hash(url))}.png"
            local_path = os.path.join(output_dir, filename)
            with open(local_path, "wb") as f:
                f.write(raw_bytes)
            return local_path
    except Exception:
        pass

    # Method 2: HTTP GET from MinIO URL
    res = requests.get(url, timeout=15)
    res.raise_for_status()
    parsed = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed.path) or f"minio_{abs(hash(url))}.png"
    local_path = os.path.join(output_dir, filename)
    with open(local_path, "wb") as f:
        f.write(res.content)
    return local_path


def query_pipeline(backend_url: str, query: str, img_cache_dir: str, top_k: int = 3) -> tuple[str, List[str]]:
    """
    Queries the RAG pipeline directly:
      1. Calls Backend API or RAG service to get Qwen-VL response and retrieved MinIO image URLs.
      2. Downloads retrieved images from MinIO into local cache.
    """
    retrieved_images = []
    response_text = ""

    # Call FastAPI backend chat endpoint
    endpoint = f"{backend_url.rstrip('/')}/api/v1/chat"
    res = requests.post(
        endpoint,
        data={"text": query},
        timeout=90,
        stream=True
    )
    res.raise_for_status()

    session_id = None
    minio_urls = []
    content_type = res.headers.get("content-type", "")

    if "text/event-stream" in content_type:
        for line in res.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            if decoded.startswith("data: "):
                data_str = decoded[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    payload = json.loads(data_str)
                    if "session_id" in payload:
                        session_id = payload["session_id"]
                    if "content" in payload:
                        response_text += payload["content"]
                    if "retrieved_images" in payload:
                        minio_urls.extend(payload["retrieved_images"])
                except Exception:
                    continue
    else:
        try:
            data = res.json()
            response_text = data.get("response") or data.get("content", "")
            session_id = data.get("session_id")
            minio_urls = data.get("retrieved_images", [])
        except Exception:
            response_text = res.text

    # Retrieve MinIO image URLs from the created session in database/API
    if session_id and not minio_urls:
        try:
            sess_res = requests.get(f"{backend_url.rstrip('/')}/api/v1/sessions/{session_id}", timeout=10)
            if sess_res.status_code == 200:
                sess_data = sess_res.json()
                for msg in sess_data.get("messages", []):
                    if msg.get("role") == "user" and msg.get("image_urls"):
                        for u in msg["image_urls"]:
                            if u not in minio_urls:
                                minio_urls.append(u)
        except Exception as e:
            pass

    if not minio_urls:
        # Check direct RAG service if backend response didn't include URLs
        try:
            from backend.services.rag_service import encode_query, search_qdrant
            query_emb = encode_query(query)
            if query_emb:
                minio_urls = search_qdrant(query_emb, top_k=top_k)
        except Exception:
            pass

    for url in minio_urls:
        try:
            local_img = download_image_from_minio(url, img_cache_dir)
            if local_img and local_img not in retrieved_images:
                retrieved_images.append(local_img)
        except Exception as e:
            print(f"[!] Failed to fetch image from MinIO ({url}): {e}")

    return response_text, retrieved_images


async def _score_with_retry(coro_func, max_retries: int = 4, base_wait: float = 6.0):
    """Retries async call with exponential backoff on HTTP 429 / rate limits (essential for Free Tier)."""
    for attempt in range(max_retries):
        try:
            return await coro_func()
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate" in err_str or "quota" in err_str or "resource_exhausted" in err_str) and attempt < max_retries - 1:
                wait_time = base_wait * (2 ** attempt)
                print(f"\n[!] Rate limit reached. Backing off for {wait_time:.1f}s (Attempt {attempt+1}/{max_retries})...")
                await asyncio.sleep(wait_time)
            else:
                raise e


async def evaluate_single_turn(
    faithfulness_metric: MultiModalFaithfulness,
    relevance_metric: MultiModalRelevance,
    query: str,
    response: str,
    context_images: List[str]
) -> Dict[str, Any]:
    """Evaluates a single VQA question against retrieved context images using Ragas."""
    # 1. MultiModalFaithfulness
    try:
        faith_res = await _score_with_retry(
            lambda: faithfulness_metric.ascore(
                response=response,
                retrieved_contexts=context_images
            )
        )
        faith_score = float(faith_res.value) if faith_res is not None and hasattr(faith_res, "value") else None
    except Exception as e:
        print(f"[!] Faithfulness eval error: {e}")
        faith_score = None

    # 2. MultiModalRelevance
    try:
        rel_res = await _score_with_retry(
            lambda: relevance_metric.ascore(
                user_input=query,
                response=response,
                retrieved_contexts=context_images
            )
        )
        rel_score = float(rel_res.value) if rel_res is not None and hasattr(rel_res, "value") else None
    except Exception as e:
        print(f"[!] Relevance eval error: {e}")
        rel_score = None

    return {
        "faithfulness": faith_score,
        "relevance": rel_score
    }


async def main_async():
    parser = argparse.ArgumentParser(description="Evaluate Scientific Document VQA using Multimodal Ragas (Pure Pipeline)")
    parser.add_argument("--test-set", type=str, default="data/test_set_100_vqa.json",
                        help="Path to JSON test set containing queries")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8000",
                        help="FastAPI Backend URL")
    parser.add_argument("--provider", type=str, default=None, choices=["openai", "gemini", "litellm"],
                        help="Cloud API provider to judge. Auto-detected if omitted.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name for judge (e.g. gpt-4o-mini, gemini-2.0-flash)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to evaluate")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index in the test set")
    parser.add_argument("--img-cache-dir", type=str, default="data/eval_page_images",
                        help="Directory to cache images downloaded from MinIO")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Output path for CSV evaluation report")
    parser.add_argument("--delay", type=float, default=None,
                        help="Delay in seconds between questions (default: 4.0s for Gemini Free Tier to stay under 15 RPM, 0.5s for others)")
    args = parser.parse_args()

    # 1. Determine Cloud Provider & Delay
    provider = args.provider
    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.getenv("LITELLM_BASE_URL"):
            provider = "litellm"
        else:
            print("[!] Error: No Cloud API key found! Please export OPENAI_API_KEY or GEMINI_API_KEY.")
            sys.exit(1)

    delay = args.delay if args.delay is not None else (4.0 if provider == "gemini" else 0.5)
    if delay > 0:
        print(f"[*] Rate-limiting delay set to {delay}s per query (safe for {provider.upper()} Free Tier).")

    # 2. Load Test Set
    if not os.path.exists(args.test_set):
        print(f"[!] Error: Test set file not found: {args.test_set}")
        sys.exit(1)

    with open(args.test_set, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    if args.start_idx > 0:
        test_samples = test_samples[args.start_idx:]
    if args.limit is not None and args.limit > 0:
        test_samples = test_samples[:args.limit]

    print(f"[*] Loaded {len(test_samples)} test questions from {args.test_set}")

    # 3. Initialize Judge LLM & Ragas Metrics
    judge_llm = init_judge_llm(provider, args.model)
    faithfulness_metric = MultiModalFaithfulness(llm=judge_llm)
    relevance_metric = MultiModalRelevance(llm=judge_llm)

    # 4. Evaluate loop
    results = []
    print(f"\n[*] Starting Multimodal Ragas Evaluation on {len(test_samples)} samples...")
    start_time = time.perf_counter()

    for idx, item in enumerate(tqdm(test_samples, desc="Evaluating Pipeline with Cloud Judge")):
        q_id = idx + 1
        query = item["query"] if isinstance(item, dict) else str(item)

        # Query live pipeline: gets response and downloads retrieved images from MinIO
        response, context_images = query_pipeline(args.backend_url, query, args.img_cache_dir)

        if not response:
            print(f"[!] Warning: No response received from pipeline for question {q_id}")
        if not context_images:
            print(f"[!] Warning: No images retrieved from MinIO for question {q_id}")

        # Run evaluation metrics
        scores = await evaluate_single_turn(
            faithfulness_metric=faithfulness_metric,
            relevance_metric=relevance_metric,
            query=query,
            response=response,
            context_images=context_images
        )

        results.append({
            "id": q_id,
            "query": query,
            "response": response,
            "num_context_images": len(context_images),
            "retrieved_images": str(context_images),
            "multimodal_faithfulness": scores["faithfulness"],
            "multimodal_relevance": scores["relevance"]
        })

        if delay > 0 and idx < len(test_samples) - 1:
            await asyncio.sleep(delay)

    elapsed = time.perf_counter() - start_time
    df = pd.DataFrame(results)

    # 5. Print Summary
    print("\n" + "=" * 60)
    print("📊 MULTIMODAL RAGAS EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated Questions : {len(df)}")
    print(f"Total Execution Time      : {elapsed:.2f}s ({elapsed / max(1, len(df)):.2f}s/sample)")
    
    mean_faith = df["multimodal_faithfulness"].dropna().mean()
    mean_rel = df["multimodal_relevance"].dropna().mean()
    
    print(f"🔸 Mean MultiModal Faithfulness: {mean_faith:.4f} (Valid: {df['multimodal_faithfulness'].count()}/{len(df)})")
    print(f"🔸 Mean MultiModal Relevance   : {mean_rel:.4f} (Valid: {df['multimodal_relevance'].count()}/{len(df)})")

    # 6. Export report
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = args.output_csv or os.path.join(output_dir, f"ragas_multimodal_eval_{timestamp}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[+] Detailed evaluation report saved to: {out_csv}")
    print("=" * 60)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
import uuid
import hashlib
import json
import base64
import urllib.parse
import logging
import time
from openai import AsyncOpenAI

from backend.db.database import get_db
from backend.db.models import ChatSession, Message, utc_now
from backend.services.storage import storage_service
from backend.services.cache_service import (
    compute_image_hash,
    check_cache_exact,
    check_cache_semantic,
    save_to_cache
)
from backend.core.config import settings
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
router = APIRouter()

_session_image_hashes: dict[str, str] = {}

openai_client = AsyncOpenAI(
    base_url=settings.LITELLM_BASE_URL,
    api_key=settings.LITELLM_API_KEY,
    max_retries=0
)

import io
from PIL import Image

def resize_image_if_needed(image_bytes: bytes, max_dimension: int = 1280) -> bytes:
    """Resizes image to max_dimension keeping aspect ratio to minimize vision tokens on fallback."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            width, height = img.size
            if max(width, height) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return buf.getvalue()
    except Exception as e:
        logger.error(f"Image resize failed: {e}")
    return image_bytes

def create_downscaled_payload(messages: list) -> list:
    """Downscales images inside message payload to max 1280px and caps total images to 3."""
    new_messages = []
    total_imgs = 0
    for msg in messages:
        if isinstance(msg.get("content"), list):
            new_content = []
            for item in msg["content"]:
                if item.get("type") == "image_url":
                    if total_imgs >= 3:
                        continue
                    total_imgs += 1
                    img_url = item.get("image_url", {}).get("url", "")
                    if img_url.startswith("data:"):
                        try:
                            header, b64_data = img_url.split(",", 1)
                            raw_bytes = base64.b64decode(b64_data)
                            resized_bytes = resize_image_if_needed(raw_bytes, max_dimension=1280)
                            new_b64 = base64.b64encode(resized_bytes).decode("utf-8")
                            new_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{new_b64}"}})
                            continue
                        except Exception as e:
                            logger.error(f"Failed to downscale image: {e}")
                    new_content.append(item)
                else:
                    new_content.append(item)
            new_messages.append({"role": msg["role"], "content": new_content})
        else:
            new_messages.append(msg)
    return new_messages

async def get_downscaled_image_payload(url: str, default_bytes: Optional[bytes] = None) -> Optional[str]:
    """
    Fetches image bytes (from MinIO or default_bytes), downscales to max 1280px (JPEG 85),
    and returns a base64 data URI to save vision tokens and VRAM.
    """
    try:
        raw_bytes = default_bytes
        if not raw_bytes:
            parsed = urllib.parse.urlparse(url)
            path = parsed.path.lstrip("/")
            parts = path.split("/", 1)
            if len(parts) == 2:
                bucket_name, object_name = parts[0], urllib.parse.unquote(parts[1])
            else:
                bucket_name, object_name = None, urllib.parse.unquote(path)
            raw_bytes = await storage_service.get_file_bytes_async(object_name, bucket_name=bucket_name)

        if raw_bytes:
            resized_bytes = resize_image_if_needed(raw_bytes, max_dimension=1280)
            b64_str = base64.b64encode(resized_bytes).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_str}"
    except Exception as e:
        logger.error(f"Failed to fetch and downscale image {url}: {e}")
    return url

@router.post("/chat")
async def chat_endpoint(
    text: str = Form(...),
    session_id: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[]),
    db: AsyncSession = Depends(get_db)
):
    if len(images) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 images allowed.")

    # 1. Generate unique session_id if not provided
    if not session_id:
        session_id = uuid.uuid4().hex

    # Ensure ChatSession exists in DB or create a new one
    session_stmt = select(ChatSession).where(ChatSession.session_id == session_id)
    session_res = await db.execute(session_stmt)
    chat_session = session_res.scalar_one_or_none()

    if not chat_session:
        title = (text[:40] + "...") if len(text) > 40 else text
        chat_session = ChatSession(
            session_id=session_id,
            title=title
        )
        db.add(chat_session)
        await db.commit()
    else:
        chat_session.updated_at = utc_now()


    # 2. Upload images to MinIO
    image_urls = []
    image_contents = []
    for img in images:
        content = await img.read()
        image_contents.append((img.filename, content, img.content_type))
        try:
            url = await storage_service.upload_file_async(
                f"{session_id}_{img.filename}",
                content,
                img.content_type
            )
            image_urls.append(url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

    # 3. Load past messages from DB
    stmt = select(Message).where(Message.session_id == session_id).order_by(Message.timestamp.asc())
    result = await db.execute(stmt)
    past_messages = result.scalars().all()

    # Determine Anchor Image Hash for cache isolation
    current_image_hash = "none"
    if image_contents:
        current_image_hash = compute_image_hash(image_contents[0][1])
        _session_image_hashes[session_id] = current_image_hash
    elif session_id in _session_image_hashes:
        current_image_hash = _session_image_hashes[session_id]
    else:
        for msg in past_messages:
            if msg.role == "user" and msg.image_urls:
                for u in msg.image_urls:
                    raw_name = u.split("/")[-1].split("?")[0]
                    object_name = urllib.parse.unquote(raw_name)
                    if session_id in object_name:
                        try:
                            img_bytes = await storage_service.get_file_bytes_async(object_name)
                            if img_bytes:
                                current_image_hash = compute_image_hash(img_bytes)
                                _session_image_hashes[session_id] = current_image_hash
                                break
                        except Exception:
                            pass
                if current_image_hash != "none":
                    break

    # 4. Check Qdrant Semantic Cache (Tier 1 Exact + Tier 2 Semantic)
    cached_result = None
    query_emb = None
    if text:
        # Tier 1: Instant Exact Match (<1ms)
        cached_result = check_cache_exact(image_hash=current_image_hash, query_text=text)

        # Tier 2: Semantic Similarity Match
        if not cached_result:
            try:
                from backend.services.rag_service import encode_query
                query_emb = encode_query(text)
                cached_result = check_cache_semantic(
                    image_hash=current_image_hash,
                    query_text=text,
                    query_emb=query_emb,
                    similarity_threshold=0.80
                )
            except Exception as e:
                logger.warning(f"Error checking semantic cache: {e}")

    # If CACHE HIT: return cached answer and retrieved pages immediately (Bypassing RAG & vLLM)
    if cached_result:
        cached_answer = cached_result["answer"]
        cached_retrieved = cached_result.get("retrieved_images", [])

        # Include cached retrieved images in message image_urls
        for u in cached_retrieved:
            if u not in image_urls:
                image_urls.append(u)

        # Save user message to DB
        new_user_msg = Message(
            session_id=session_id,
            role="user",
            content=text,
            image_urls=image_urls
        )
        db.add(new_user_msg)

        # Save assistant message to DB
        new_assistant_msg = Message(
            session_id=session_id,
            role="assistant",
            content=cached_answer,
            image_urls=[]
        )
        db.add(new_assistant_msg)
        await db.commit()

        async def cached_stream_generator():
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'content': cached_answer})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(cached_stream_generator(), media_type="text/event-stream")

    # 5. CACHE MISS: Search Qdrant for relevant textbook pages
    try:
        from backend.services.rag_service import encode_query, search_qdrant
        if text:
            if not query_emb:
                query_emb = encode_query(text)
            rag_image_urls = search_qdrant(query_emb, top_k=3)
            # Add unique URLs
            for url in rag_image_urls:
                if url not in image_urls:
                    image_urls.append(url)
    except Exception as e:
        logger.error(f"RAG search failed: {e}")

    # 6. Prepare payload for LiteLLM / vLLM (OpenAI Vision format) with max 4 images
    messages_payload = [{"role": "system", "content": settings.SYSTEM_PROMPT}]
    seen_image_keys = set()
    total_images_included = 0
    MAX_VISION_IMAGES = 4  # vLLM maximum vision image limit
    
    for msg in past_messages:
        content_payload = [{"type": "text", "text": msg.content}]
        if msg.image_urls:
            for url in msg.image_urls:
                raw_name = url.split("/")[-1].split("?")[0]
                object_name = urllib.parse.unquote(raw_name)
                
                # Classification: If URL contains session_id, it is a direct user-uploaded anchor image.
                # Otherwise, it is an ephemeral RAG retrieved image -> omit past RAG images to make room for current turn
                is_anchor = session_id in object_name
                if not is_anchor:
                    continue
                
                # Retain unique anchor images up to the 4-image cap
                if object_name in seen_image_keys or total_images_included >= MAX_VISION_IMAGES:
                    continue
                seen_image_keys.add(object_name)
                
                img_url_to_send = await get_downscaled_image_payload(url)
                content_payload.append({"type": "image_url", "image_url": {"url": img_url_to_send}})
                total_images_included += 1
        messages_payload.append({"role": msg.role, "content": content_payload})

    # Add current user message
    current_content = [{"type": "text", "text": text}]
    for i, (filename, content, content_type) in enumerate(image_contents):
        object_name = f"{session_id}_{filename}"
        if object_name in seen_image_keys or total_images_included >= MAX_VISION_IMAGES:
            continue
        seen_image_keys.add(object_name)
        
        url = image_urls[i]
        img_payload = await get_downscaled_image_payload(url, default_bytes=content)
        current_content.append({"type": "image_url", "image_url": {"url": img_payload}})
        total_images_included += 1
    
    # Add RAG images to current payload (up to the 4-image cap)
    for url in image_urls[len(image_contents):]:
        if total_images_included >= MAX_VISION_IMAGES:
            break
        raw_name = url.split("/")[-1].split("?")[0]
        object_name = urllib.parse.unquote(raw_name)
        if object_name in seen_image_keys:
            continue
        seen_image_keys.add(object_name)
        
        img_url_to_send = await get_downscaled_image_payload(url)
        current_content.append({"type": "image_url", "image_url": {"url": img_url_to_send}})
        total_images_included += 1

    messages_payload.append({"role": "user", "content": current_content})

    # Save user message to DB
    new_user_msg = Message(
        session_id=session_id,
        role="user",
        content=text,
        image_urls=image_urls
    )
    db.add(new_user_msg)
    await db.commit()


    # 5. Call LiteLLM Gateway using OpenAI SDK
    async def stream_generator():
        assistant_reply = ""
        # Send session_id metadata in the first SSE event
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        
        llm_start_time = time.perf_counter()
        first_token_received = False
        model_used = "openai/qwen-vl-vqa"
        
        try:
            stream = await openai_client.chat.completions.create(
                model=model_used,
                messages=messages_payload,
                stream=True,
                user=session_id,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content or ""
                    if delta_content:
                        if not first_token_received:
                            first_token_received = True
                            ttft_ms = (time.perf_counter() - llm_start_time) * 1000
                            logger.info(f"[LATENCY] LLM Time-To-First-Token (TTFT) [{model_used}]: {ttft_ms:.2f}ms")
                        assistant_reply += delta_content
                        yield f"data: {json.dumps({'content': delta_content})}\n\n"
            total_duration_ms = (time.perf_counter() - llm_start_time) * 1000
            logger.info(f"[LATENCY] LLM Total Generation Time [{model_used}]: {total_duration_ms:.2f}ms (Response length: {len(assistant_reply)} chars)")
        except Exception as primary_err:
            logger.warning(f"Primary vLLM call failed ({primary_err}), attempting downscaled fallback to Groq...")
            model_used = "groq/qwen/qwen3.6-27b"
            fallback_start_time = time.perf_counter()
            first_token_received = False
            try:
                downscaled_payload = create_downscaled_payload(messages_payload)
                fallback_stream = await openai_client.chat.completions.create(
                    model=model_used,
                    messages=downscaled_payload,
                    stream=True,
                    user=session_id,
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS
                )
                async for chunk in fallback_stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content or ""
                        if delta_content:
                            if not first_token_received:
                                first_token_received = True
                                ttft_ms = (time.perf_counter() - fallback_start_time) * 1000
                                logger.info(f"[LATENCY] Fallback LLM TTFT [{model_used}]: {ttft_ms:.2f}ms")
                            assistant_reply += delta_content
                            yield f"data: {json.dumps({'content': delta_content})}\n\n"
                total_duration_ms = (time.perf_counter() - fallback_start_time) * 1000
                logger.info(f"[LATENCY] Fallback LLM Total Generation Time [{model_used}]: {total_duration_ms:.2f}ms (Response length: {len(assistant_reply)} chars)")
            except Exception as fallback_err:
                logger.error(f"Fallback also failed: {fallback_err}")
                yield f"data: {json.dumps({'error': str(fallback_err)})}\n\n"
        
        # Save assistant reply to DB
        if assistant_reply:
            new_assistant_msg = Message(
                session_id=session_id,
                role="assistant",
                content=assistant_reply,
                image_urls=[]
            )
            db.add(new_assistant_msg)
            await db.commit()

            # Save Q&A to Qdrant Semantic Cache asynchronously
            if text and query_emb:
                try:
                    save_to_cache(
                        image_hash=current_image_hash,
                        query_text=text,
                        query_emb=query_emb,
                        answer=assistant_reply,
                        retrieved_images=image_urls[len(image_contents):]
                    )
                except Exception as cache_save_err:
                    logger.warning(f"Failed to save to Qdrant cache: {cache_save_err}")
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

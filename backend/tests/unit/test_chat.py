import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace
import json
from backend.core.config import settings

def make_chunk(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )

async def async_chunk_generator(chunks):
    for c in chunks:
        yield make_chunk(c)

def make_mock_create(chunks):
    async def _mock_create(**kwargs):
        return async_chunk_generator(chunks)
    return _mock_create

@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "db" in response.json()

@pytest.mark.asyncio
@patch("backend.services.storage.storage_service.upload_file")
async def test_chat_validation_max_images(mock_upload, client: AsyncClient):
    mock_upload.return_value = "http://mock-minio/image.jpg"
    files = [("images", (f"test{i}.jpg", b"fake_data", "image/jpeg")) for i in range(4)]
    data = {"text": "Too many images"}
    
    response = await client.post("/api/v1/chat", data=data, files=files)
    assert response.status_code == 400
    assert "Maximum 3 images allowed" in response.text

@pytest.mark.asyncio
@patch("backend.services.storage.storage_service.upload_file")
@patch("backend.api.routes.chat.openai_client.chat.completions.create")
async def test_chat_happy_path_base64_fallback(mock_create, mock_upload, client: AsyncClient):
    mock_upload.return_value = "http://mock-minio/image.jpg"
    mock_create.side_effect = make_mock_create(["Hello", " World"])
    
    settings.MINIO_PUBLIC_ENDPOINT = ""
    files = [("images", ("test.jpg", b"fake_data", "image/jpeg"))]
    data = {"text": "What is this?"}
    
    response = await client.post("/api/v1/chat", data=data, files=files)
    assert response.status_code == 200
    
    content = ""
    received_session_id = None
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str != "[DONE]":
                parsed = json.loads(data_str)
                if "session_id" in parsed:
                    received_session_id = parsed["session_id"]
                if "content" in parsed:
                    content += parsed["content"]
    
    assert content == "Hello World"
    assert received_session_id is not None

@pytest.mark.asyncio
@patch("backend.services.storage.storage_service.upload_file")
@patch("backend.api.routes.chat.openai_client.chat.completions.create")
async def test_chat_happy_path_public_url(mock_create, mock_upload, client: AsyncClient):
    mock_upload.return_value = "http://public-minio.com/bucket/image.jpg"
    mock_create.side_effect = make_mock_create(["From URL"])
    
    settings.MINIO_PUBLIC_ENDPOINT = "http://public-minio.com"
    files = [("images", ("test.jpg", b"fake_data", "image/jpeg"))]
    data = {"text": "What is this?"}
    
    response = await client.post("/api/v1/chat", data=data, files=files)
    assert response.status_code == 200
    
    content = ""
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str != "[DONE]":
                parsed = json.loads(data_str)
                if "content" in parsed:
                    content += parsed["content"]
    assert content == "From URL"
    settings.MINIO_PUBLIC_ENDPOINT = ""

@pytest.mark.asyncio
@patch("backend.api.routes.chat.openai_client.chat.completions.create")
async def test_chat_text_only(mock_create, client: AsyncClient):
    mock_create.side_effect = make_mock_create(["Text answer"])
    
    data = {"text": "Just a text question"}
    response = await client.post("/api/v1/chat", data=data)
    assert response.status_code == 200
    
    content = ""
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str != "[DONE]":
                parsed = json.loads(data_str)
                if "content" in parsed:
                    content += parsed["content"]
    assert content == "Text answer"

@pytest.mark.asyncio
@patch("backend.api.routes.chat.openai_client.chat.completions.create")
async def test_chat_subsequent_turn_with_session_id(mock_create, client: AsyncClient):
    mock_create.side_effect = make_mock_create(["Turn 2 answer"])
    
    data = {"text": "Follow-up question", "session_id": "custom_session_123"}
    response = await client.post("/api/v1/chat", data=data)
    assert response.status_code == 200
    
    received_session_id = None
    content = ""
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str != "[DONE]":
                parsed = json.loads(data_str)
                if "session_id" in parsed:
                    received_session_id = parsed["session_id"]
                if "content" in parsed:
                    content += parsed["content"]
                    
    assert content == "Turn 2 answer"
    assert received_session_id == "custom_session_123"

@pytest.mark.asyncio
@patch("backend.api.routes.chat.openai_client.chat.completions.create")
async def test_chat_openai_error(mock_create, client: AsyncClient):
    mock_create.side_effect = Exception("LiteLLM connection error")
    
    data = {"text": "Trigger error"}
    response = await client.post("/api/v1/chat", data=data)
    assert response.status_code == 200
    
    events = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            events.append(line[6:])
    
    assert len(events) >= 2
    parsed_err = json.loads(events[1])
    assert "error" in parsed_err
    assert "LiteLLM connection error" in parsed_err["error"]

@pytest.mark.asyncio
@patch("backend.services.storage.storage_service.get_file_bytes_async", return_value=b"fake_downscaled_bytes")
@patch("backend.api.routes.chat.openai_client.chat.completions.create")
async def test_chat_anchor_image_kept_and_rag_dropped(mock_create, mock_get_bytes, client: AsyncClient):
    mock_create.side_effect = make_mock_create(["Follow-up reply"])
    session_id = "test_anchor_sess"
    
    from backend.tests.conftest import TestingSessionLocal
    from backend.db.models import ChatSession, Message
    from datetime import datetime, timezone
    
    async with TestingSessionLocal() as db:
        sess = ChatSession(session_id=session_id, title="Test Anchor", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        msg1 = Message(
            session_id=session_id,
            role="user",
            content="Turn 1 question",
            image_urls=[
                f"http://minio/vqa-images/{session_id}_user_chart.png",
                "http://minio/scientific-images/pdf-images/page_10.png"
            ]
        )
        msg2 = Message(session_id=session_id, role="assistant", content="Turn 1 answer")
        db.add_all([sess, msg1, msg2])
        await db.commit()
        
    data = {"text": "What does the 3rd column mean?", "session_id": session_id}
    response = await client.post("/api/v1/chat", data=data)
    assert response.status_code == 200
    
    mock_create.assert_called_once()
    _, kwargs = mock_create.call_args
    messages = kwargs["messages"]
    
    turn1_payload = messages[1]["content"]
    img_urls = [item["image_url"]["url"] for item in turn1_payload if item.get("type") == "image_url"]
    
    # Verify exactly 1 image was kept from Turn 1 (the Anchor Image, NOT the RAG image)
    assert len(img_urls) == 1

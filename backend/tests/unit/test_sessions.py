import pytest
from httpx import AsyncClient
from backend.db.models import ChatSession, Message
from backend.tests.conftest import TestingSessionLocal
from datetime import datetime, timezone

@pytest.mark.asyncio
async def test_list_sessions_empty(client: AsyncClient):
    response = await client.get("/api/v1/sessions")
    assert response.status_code == 200
    assert response.json() == []

@pytest.mark.asyncio
async def test_list_and_get_sessions(client: AsyncClient):
    async with TestingSessionLocal() as db:
        session1 = ChatSession(session_id="sess_1", title="First Paper QA", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        msg1 = Message(session_id="sess_1", role="user", content="Explain Figure 1")
        msg2 = Message(session_id="sess_1", role="assistant", content="Figure 1 shows accuracy vs epoch.")
        
        session2 = ChatSession(session_id="sess_2", title="Second Paper QA", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        
        db.add_all([session1, msg1, msg2, session2])
        await db.commit()

    # Test List
    response = await client.get("/api/v1/sessions")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    sess1_data = next(s for s in data if s["session_id"] == "sess_1")
    assert sess1_data["title"] == "First Paper QA"
    assert sess1_data["message_count"] == 2

    # Test Get Detail
    detail_res = await client.get("/api/v1/sessions/sess_1")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["session_id"] == "sess_1"
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["content"] == "Explain Figure 1"
    assert detail["messages"][1]["content"] == "Figure 1 shows accuracy vs epoch."

@pytest.mark.asyncio
async def test_get_session_not_found(client: AsyncClient):
    response = await client.get("/api/v1/sessions/non_existent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"

@pytest.mark.asyncio
async def test_delete_session(client: AsyncClient):
    async with TestingSessionLocal() as db:
        session = ChatSession(session_id="to_delete", title="Temp Session", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        msg = Message(session_id="to_delete", role="user", content="Temporary message")
        db.add_all([session, msg])
        await db.commit()


    del_res = await client.delete("/api/v1/sessions/to_delete")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # Verify session is deleted
    get_res = await client.get("/api/v1/sessions/to_delete")
    assert get_res.status_code == 404

@pytest.mark.asyncio
async def test_delete_session_not_found(client: AsyncClient):
    response = await client.delete("/api/v1/sessions/non_existent")
    assert response.status_code == 404

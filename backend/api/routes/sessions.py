from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from backend.db.database import get_db
from backend.db.models import ChatSession, Message
from backend.schemas.session import SessionSummary, SessionDetail, MessageDetail

router = APIRouter()


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """Lists all chat sessions sorted by latest activity."""
    stmt = select(ChatSession).order_by(ChatSession.updated_at.desc())
    result = await db.execute(stmt)
    sessions = result.scalars().all()
    
    response = []
    for s in sessions:
        msg_stmt = select(Message).where(Message.session_id == s.session_id)
        msg_result = await db.execute(msg_stmt)
        msgs = msg_result.scalars().all()
        response.append(SessionSummary(
            session_id=s.session_id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=len(msgs)
        ))
    return response

@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieves session details and its full conversation history."""
    stmt = select(ChatSession).where(ChatSession.session_id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    msg_stmt = select(Message).where(Message.session_id == session_id).order_by(Message.timestamp.asc())
    msg_result = await db.execute(msg_stmt)
    messages = msg_result.scalars().all()

    return SessionDetail(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[MessageDetail.model_validate(m) for m in messages]
    )

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """Deletes a chat session and cascades all messages."""
    stmt = select(ChatSession).where(ChatSession.session_id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await db.delete(session)
    await db.commit()
    return {"status": "success", "message": f"Session {session_id} deleted"}

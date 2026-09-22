from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from backend.db.database import Base
from datetime import datetime, timezone

def utc_now():
    """Return naive UTC datetime matching PostgreSQL TIMESTAMP WITHOUT TIME ZONE."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.timestamp.asc()")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True, nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    image_urls = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=utc_now)

    session = relationship("ChatSession", back_populates="messages")



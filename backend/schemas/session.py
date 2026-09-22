from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class MessageDetail(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    image_urls: Optional[List[str]] = None
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)

class SessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

    model_config = ConfigDict(from_attributes=True)

class SessionDetail(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageDetail]

    model_config = ConfigDict(from_attributes=True)

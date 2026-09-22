from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import chat, sessions
from backend.db.database import engine, Base
import asyncio
import logging
from contextlib import asynccontextmanager
from sqlalchemy import text

# Structured Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application, connecting to DB and running migrations...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        logger.warning(f"Could not connect to database on startup: {e}")
    try:
        from backend.services.cache_service import ensure_cache_collection
        ensure_cache_collection()
    except Exception as e:
        logger.warning(f"Could not initialize cache collection: {e}")
    yield
    logger.info("Shutting down application...")

app = FastAPI(title="Scientific Document VQA API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, set this from environment variables
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])


@app.get("/health")
async def health_check():
    # Deep health check checking database connectivity
    db_status = "ok"
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        db_status = f"warning: {str(e)}"
        
    return {"status": "ok", "db": db_status}

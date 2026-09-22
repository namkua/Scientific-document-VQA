from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/vqa_db"
    
    # MinIO
    MINIO_URL: str = "localhost:9000"
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "vqa-images"
    MINIO_PUBLIC_ENDPOINT: str = "" # Set this to public IP/Domain in production
    
    # AI Gateway (LiteLLM)
    LITELLM_BASE_URL: str = "http://localhost:4000/v1"
    LITELLM_API_KEY: str = "sk-dummy"
    
    # Embedder Microservice (ColQwen2.5 on Vast.ai or internal)
    EMBEDDER_URL: str = "http://localhost:8000"
    EMBEDDER_API_KEY: str = ""
    
    # LLM Inference Hyperparameters
    SYSTEM_PROMPT: str = (
        "Bạn là trợ lý AI chuyên gia về phân tích tài liệu khoa học, bài báo nghiên cứu, "
        "biểu đồ, đồ thị, bảng biểu, sơ đồ và phương trình toán học. "
        "Hãy cung cấp câu trả lời chính xác, chặt chẽ, có cấu trúc rõ ràng và luôn phản hồi bằng tiếng Việt "
        "dựa trên tài liệu và hình ảnh được cung cấp. "
        "Khi trích xuất dữ liệu hoặc đọc số liệu từ biểu đồ/bảng biểu, hãy đảm bảo độ chính xác tuyệt đối. "
        "Nếu một chi tiết nào đó không thể nhìn rõ hoặc không xuất hiện trong tài liệu, hãy nêu rõ ràng."
    )
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2048
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()


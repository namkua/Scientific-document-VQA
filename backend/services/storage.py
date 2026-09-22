from minio import Minio
from minio.error import S3Error
from backend.core.config import settings
from datetime import timedelta
import io
import asyncio

class MinioService:
    def __init__(self):
        self.client = Minio(
            settings.MINIO_URL,
            access_key=settings.MINIO_ROOT_USER,
            secret_key=settings.MINIO_ROOT_PASSWORD,
            secure=False
        )
        self.bucket_name = settings.MINIO_BUCKET_NAME
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except Exception as e:
            pass

    def upload_file(self, object_name: str, file_data: bytes, content_type: str) -> str:
        self.client.put_object(
            self.bucket_name,
            object_name,
            io.BytesIO(file_data),
            len(file_data),
            content_type=content_type
        )
        
        if settings.MINIO_PUBLIC_ENDPOINT:
            url = f"{settings.MINIO_PUBLIC_ENDPOINT}/{self.bucket_name}/{object_name}"
        else:
            url = self.client.get_presigned_url(
                "GET",
                self.bucket_name,
                object_name,
                expires=timedelta(days=1),
            )
        return url

    async def upload_file_async(self, object_name: str, file_data: bytes, content_type: str) -> str:
        return await asyncio.to_thread(self.upload_file, object_name, file_data, content_type)

    def get_file_bytes(self, object_name: str, bucket_name: str = None) -> bytes:
        target_bucket = bucket_name or self.bucket_name
        response = self.client.get_object(target_bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    async def get_file_bytes_async(self, object_name: str, bucket_name: str = None) -> bytes:
        return await asyncio.to_thread(self.get_file_bytes, object_name, bucket_name)

storage_service = MinioService()


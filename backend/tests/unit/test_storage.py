import pytest
from unittest.mock import patch, MagicMock
from backend.services.storage import MinioService
from backend.core.config import settings

@patch("backend.services.storage.Minio")
def test_upload_file_no_public_endpoint(mock_minio_class):
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = True
    mock_client.get_presigned_url.return_value = "http://presigned-url"
    mock_minio_class.return_value = mock_client
    
    settings.MINIO_PUBLIC_ENDPOINT = ""
    service = MinioService()
    url = service.upload_file("test.jpg", b"fake", "image/jpeg")
    
    assert url == "http://presigned-url"
    mock_client.put_object.assert_called_once()
    mock_client.get_presigned_url.assert_called_once()

@patch("backend.services.storage.Minio")
def test_upload_file_with_public_endpoint(mock_minio_class):
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = True
    mock_minio_class.return_value = mock_client
    
    settings.MINIO_PUBLIC_ENDPOINT = "http://public.com"
    service = MinioService()
    url = service.upload_file("test.jpg", b"fake", "image/jpeg")
    
    assert url == f"http://public.com/{settings.MINIO_BUCKET_NAME}/test.jpg"
    mock_client.put_object.assert_called_once()
    mock_client.get_presigned_url.assert_not_called()
    
    settings.MINIO_PUBLIC_ENDPOINT = "" # reset

@pytest.mark.asyncio
@patch("backend.services.storage.Minio")
async def test_upload_file_async(mock_minio_class):
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = True
    mock_client.get_presigned_url.return_value = "http://presigned-async"
    mock_minio_class.return_value = mock_client
    
    settings.MINIO_PUBLIC_ENDPOINT = ""
    service = MinioService()
    url = await service.upload_file_async("test_async.jpg", b"fake_async", "image/jpeg")
    
    assert url == "http://presigned-async"
    mock_client.put_object.assert_called_once()

@pytest.mark.asyncio
@patch("backend.services.storage.Minio")
async def test_get_file_bytes_async(mock_minio_class):
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = True
    mock_response = MagicMock()
    mock_response.read.return_value = b"image_raw_bytes"
    mock_client.get_object.return_value = mock_response
    mock_minio_class.return_value = mock_client
    
    service = MinioService()
    data = await service.get_file_bytes_async("doc.png")
    
    assert data == b"image_raw_bytes"
    mock_client.get_object.assert_called_once()
    mock_response.close.assert_called_once()


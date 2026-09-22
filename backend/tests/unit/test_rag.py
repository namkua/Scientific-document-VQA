import pytest
from unittest.mock import patch, MagicMock
from backend.services.rag_service import encode_query, search_qdrant

def test_encode_query_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "colqwen": [[0.1, 0.2]],
        "bm25": {"indices": [1, 2], "values": [0.5, 0.8]}
    }
    
    with patch("requests.post", return_value=mock_resp) as mock_post:
        with patch("backend.services.rag_service.EMBEDDER_API_KEY", "test-secret-key"):
            result = encode_query("Đồ thị liên thông")
            
            assert "colqwen" in result
            assert result["colqwen"] == [[0.1, 0.2]]
            assert result["bm25"]["indices"] == [1, 2]
            
            # Verify X-API-Key header was sent
            mock_post.assert_called_once()
            _, kwargs = mock_post.call_args
            assert kwargs["headers"]["X-API-Key"] == "test-secret-key"

def test_encode_query_failure_fallback():
    # When embedder service is unreachable or returns 500, encode_query must not crash
    with patch("requests.post", side_effect=Exception("Connection refused")):
        result = encode_query("Tìm kiếm lỗi mạng")
        assert result == {"colqwen": [], "bm25": {"indices": [], "values": []}}

def test_search_qdrant_empty_vectors():
    # If vectors are empty, search_qdrant must immediately return empty list without querying DB
    result = search_qdrant({"colqwen": [], "bm25": {"indices": [], "values": []}})
    assert result == []

def test_search_qdrant_diffusion_ranking():
    # Mock Qdrant query_points returning page 5 with base rank
    mock_point = MagicMock()
    mock_point.payload = {
        "page_number": 5,
        "page": 5,
        "document_name": "Toan_roi_rac.pdf",
        "total_pages": 100,
        "image_url": "http://minio/pdf-images/page_5.png"
    }
    
    mock_search_result = MagicMock()
    mock_search_result.points = [mock_point]
    
    with patch("backend.services.rag_service.qdrant_client.query_points", return_value=mock_search_result):
        with patch("backend.services.rag_service.qdrant_client.scroll", return_value=([], None)):
            urls = search_qdrant({
                "colqwen": [[0.1, 0.2]],
                "bm25": {"indices": [1], "values": [0.5]}
            }, top_k=1)
            
            assert len(urls) == 1
            assert urls[0] == "http://minio/pdf-images/page_5.png"

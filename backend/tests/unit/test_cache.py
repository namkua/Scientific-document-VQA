import pytest
from unittest.mock import patch, MagicMock
from backend.services.cache_service import (
    compute_image_hash,
    normalize_text,
    check_cache_exact,
    check_cache_semantic,
    save_to_cache
)

def test_compute_image_hash():
    # Empty bytes returns 'none'
    assert compute_image_hash(b"") == "none"
    assert compute_image_hash(None) == "none"
    
    # Consistent SHA-256 for non-empty bytes
    h1 = compute_image_hash(b"test_image_data")
    h2 = compute_image_hash(b"test_image_data")
    assert h1 == h2
    assert len(h1) == 64

def test_normalize_text():
    raw = "  Thuật   TOÁN Cây   khung  "
    assert normalize_text(raw) == "thuật toán cây khung"

def test_check_cache_exact_hit():
    mock_point = MagicMock()
    mock_point.payload = {
        "answer": "Đây là câu trả lời đã cache",
        "retrieved_images": ["http://localhost:9000/images/page_1.png"],
        "query_text": "Thuật toán cây khung"
    }

    with patch("backend.services.cache_service.qdrant_client.scroll", return_value=[[mock_point], None]):
        result = check_cache_exact("test_hash_123", "Thuật toán cây khung")
        assert result is not None
        assert result["match_type"] == "exact"
        assert result["answer"] == "Đây là câu trả lời đã cache"
        assert result["retrieved_images"] == ["http://localhost:9000/images/page_1.png"]

def test_check_cache_exact_miss():
    with patch("backend.services.cache_service.qdrant_client.scroll", return_value=[[], None]):
        result = check_cache_exact("test_hash_123", "Câu hỏi chưa từng xuất hiện")
        assert result is None

def test_check_cache_semantic_hit():
    mock_hit = MagicMock()
    mock_hit.score = 0.95
    mock_hit.payload = {
        "answer": "Câu trả lời semantic cache",
        "retrieved_images": ["http://localhost:9000/images/page_88.png"],
        "query_text": "cây của đồ thị là như nào"
    }
    mock_search_res = MagicMock()
    mock_search_res.points = [mock_hit]

    query_emb = {
        "colqwen": [[0.1, 0.2]],
        "bm25": {"indices": [1, 2], "values": [0.5, 0.8]}
    }

    with patch("backend.services.cache_service.qdrant_client.query_points", return_value=mock_search_res):
        result = check_cache_semantic("test_hash_123", "thế nào là cây khung", query_emb, similarity_threshold=0.90)
        assert result is not None
        assert result["match_type"] == "semantic"
        assert result["answer"] == "Câu trả lời semantic cache"
        assert result["score"] == 0.95

def test_save_to_cache():
    query_emb = {
        "colqwen": [[0.1, 0.2]],
        "bm25": {"indices": [1, 2], "values": [0.5, 0.8]}
    }

    with patch("backend.services.cache_service.qdrant_client.upsert") as mock_upsert:
        save_to_cache(
            image_hash="hash_abc",
            query_text="Cây khung là gì?",
            query_emb=query_emb,
            answer="Cây khung là đồ thị con...",
            retrieved_images=["page_1.png"]
        )
        mock_upsert.assert_called_once()
        _, kwargs = mock_upsert.call_args
        assert kwargs["collection_name"] == "qa_semantic_cache"
        points = kwargs["points"]
        assert len(points) == 1
        assert points[0].payload["image_hash"] == "hash_abc"
        assert points[0].payload["answer"] == "Cây khung là đồ thị con..."

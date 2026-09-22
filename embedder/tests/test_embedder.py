import os
import io
import sys
import base64
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError
from PIL import Image

# Ensure embedder directory is in sys.path
EMBEDDER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if EMBEDDER_DIR not in sys.path:
    sys.path.insert(0, EMBEDDER_DIR)

# Ensure heavy model loading is skipped during unit testing
os.environ["SKIP_MODEL_LOAD"] = "1"

import main
from schemas import (
    QueryEmbedRequest,
    QueryEmbedResponse,
    ImageEmbedRequest,
    ImageEmbedResponse,
    SparseVector,
)
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create FastAPI test client with clean environment."""
    original_api_key = main.API_KEY
    main.API_KEY = ""
    with TestClient(main.app) as test_client:
        yield test_client
    main.API_KEY = original_api_key


@pytest.fixture
def sample_base64_image():
    """Generate a tiny 1x1 PNG image encoded in base64."""
    img = Image.new("RGB", (1, 1), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ==========================================
# 1. Pydantic Schema Validation Tests
# ==========================================

def test_query_embed_request_valid():
    req = QueryEmbedRequest(text="What is a binary search tree?")
    assert req.text == "What is a binary search tree?"


def test_query_embed_request_empty_rejected():
    with pytest.raises(ValidationError):
        QueryEmbedRequest(text="")


def test_image_embed_request_valid(sample_base64_image):
    req = ImageEmbedRequest(images_base64=[sample_base64_image], texts=["Sample OCR text"])
    assert len(req.images_base64) == 1
    assert req.texts == ["Sample OCR text"]


def test_image_embed_request_empty_list_rejected():
    with pytest.raises(ValidationError):
        ImageEmbedRequest(images_base64=[])


def test_sparse_vector_schema():
    sp = SparseVector(indices=[1, 45, 99], values=[0.5, 1.2, 0.8])
    assert sp.indices == [1, 45, 99]
    assert len(sp.values) == 3


# ==========================================
# 2. Endpoint Tests: Health Check
# ==========================================

def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "device" in data
    assert "auth_enabled" in data


# ==========================================
# 3. Security & Authentication Tests
# ==========================================

def test_api_key_auth_enforcement(client):
    main.API_KEY = "super-secret-key"
    try:
        # Request without header should be rejected
        resp_no_key = client.post("/embed", json={"text": "test query"})
        assert resp_no_key.status_code == 401

        # Request with incorrect key should be rejected
        resp_wrong_key = client.post(
            "/embed",
            json={"text": "test query"},
            headers={"X-API-Key": "wrong-key"}
        )
        assert resp_wrong_key.status_code == 401

        # Request with correct key should pass auth (though may fail 503 if models are None)
        main.model = None
        resp_valid_key = client.post(
            "/embed",
            json={"text": "test query"},
            headers={"X-API-Key": "super-secret-key"}
        )
        assert resp_valid_key.status_code == 503
    finally:
        main.API_KEY = ""


# ==========================================
# 4. Mocked Inference Tests
# ==========================================

def test_embed_query_mocked_success(client):
    """Test POST /embed returns expected multi-vector and sparse representations."""
    # Setup mock model and processor
    mock_model = MagicMock()
    mock_emb_tensor = MagicMock()
    mock_emb_tensor.to.return_value.to.return_value.numpy.return_value.tolist.return_value = [[0.1, 0.2] * 64]
    mock_model.return_value = [mock_emb_tensor]

    mock_processor = MagicMock()
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_processor.process_queries.return_value = mock_inputs

    # Setup mock BM25 sparse model
    mock_bm25 = MagicMock()
    mock_sparse_res = MagicMock()
    mock_sparse_res.indices.tolist.return_value = [10, 20]
    mock_sparse_res.values.tolist.return_value = [1.5, 2.0]
    mock_bm25.embed.return_value = [mock_sparse_res]

    main.model = mock_model
    main.processor = mock_processor
    main.bm25_model = mock_bm25

    try:
        response = client.post("/embed", json={"text": "Connected undirected graph"})
        assert response.status_code == 200
        data = response.json()
        assert "colqwen" in data
        assert "bm25" in data
        assert data["bm25"]["indices"] == [10, 20]
        assert data["bm25"]["values"] == [1.5, 2.0]
    finally:
        main.model, main.processor, main.bm25_model = None, None, None


def test_embed_images_mocked_success(client, sample_base64_image):
    """Test POST /embed-images parses base64 and returns batch embeddings."""
    mock_model = MagicMock()
    mock_emb_tensor = MagicMock()
    mock_emb_tensor.to.return_value.to.return_value.numpy.return_value.tolist.return_value = [[0.3, 0.4] * 64]
    mock_model.return_value = [mock_emb_tensor]

    mock_processor = MagicMock()
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_processor.process_images.return_value = mock_inputs

    mock_bm25 = MagicMock()
    mock_sparse_res = MagicMock()
    mock_sparse_res.indices.tolist.return_value = [100]
    mock_sparse_res.values.tolist.return_value = [3.2]
    mock_bm25.embed.return_value = [mock_sparse_res]

    main.model = mock_model
    main.processor = mock_processor
    main.bm25_model = mock_bm25

    try:
        response = client.post(
            "/embed-images",
            json={
                "images_base64": [sample_base64_image],
                "texts": ["Page text content"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "colqwen" in data
        assert len(data["colqwen"]) == 1
        assert "bm25" in data
        assert len(data["bm25"]) == 1
        assert data["bm25"][0]["indices"] == [100]
    finally:
        main.model, main.processor, main.bm25_model = None, None, None


def test_embed_images_invalid_base64(client):
    """Test POST /embed-images handles corrupted base64 gracefully with 400."""
    main.model = MagicMock()
    main.processor = MagicMock()
    main.bm25_model = MagicMock()

    try:
        response = client.post(
            "/embed-images",
            json={"images_base64": ["not-a-valid-base64-image"]}
        )
        assert response.status_code == 400
        assert "Failed to decode image" in response.json()["detail"]
    finally:
        main.model, main.processor, main.bm25_model = None, None, None

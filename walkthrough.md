# Walkthrough: Triển khai Chatbot RAG Khoa học

Dự án đã được cập nhật thành công theo các milestone yêu cầu. Dưới đây là tổng kết các thay đổi và hướng dẫn chạy:

## 1. Cập nhật Infrastructure
- Đã thêm service **Qdrant** vào `docker-compose.yml` (port 6333).
- Đã cập nhật `backend/requirements.txt` với các thư viện cần thiết: `qdrant-client`, `colpali-engine`, `torch`, `transformers`, `pdf2image`, `googlesearch-python`.

## 2. Scraper (Thu thập dữ liệu)
- Đã tạo `scripts/scraper.py` để tìm kiếm và tải file PDF Giáo trình Cấu trúc dữ liệu và giải thuật PTIT từ Google.
- File PDF sẽ được tự động lưu vào thư mục `data/`.

## 3. Ingestion Pipeline
- Đã tạo `scripts/ingest_colqwen.py` sử dụng thư viện `colpali-engine` với model `vidore/colqwen2.5-v0.2`.
- Script tự động đọc PDF, chuyển đổi từng trang thành hình ảnh (bằng `pdf2image`), upload lên **MinIO**, trích xuất vector embedding và đẩy lên **Qdrant** (collection `scientific_documents`).

## 4. Tích hợp Backend API
- Đã tạo `backend/services/rag_service.py` chứa hàm `encode_query` và `search_qdrant`.
- Đã chỉnh sửa luồng chat tại `backend/api/routes/chat.py`:
  - Mã hoá câu hỏi của người dùng thành vector.
  - Tìm kiếm Top-2 hình ảnh trang giáo trình liên quan nhất từ Qdrant.
  - Cập nhật payload, nối các ảnh kết quả RAG (chuyển đổi base64 nếu cần) vào prompt gửi tới **Qwen-VL** qua vLLM/LiteLLM.

## 5. Hướng dẫn chạy thử nghiệm (End-to-End)
Do môi trường Docker daemon chưa sẵn sàng ở thời điểm hiện tại, bạn có thể khởi chạy bằng tay qua các bước sau:
1. Bật Docker Desktop.
2. Chạy cơ sở hạ tầng: `docker compose up -d`
3. Cài đặt dependencies: `pip install -r backend/requirements.txt`
4. Chạy scraper: `python scripts/scraper.py`
5. Chạy ingestion: `python scripts/ingest_colqwen.py`
6. Khởi động FastAPI backend và chat thử: "Cây nhị phân tìm kiếm là gì?"

Quá trình nâng cấp đã hoàn tất!

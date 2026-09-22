import os
import io
import uuid
import base64
import argparse
import requests
from PIL import Image
from minio import Minio
from qdrant_client import QdrantClient, models
import pypdfium2 as pdfium

def load_pdf_pages(pdf_path: str, dpi: int = 150):
    """
    Extract list of PIL images and text per PDF page at DPI=150 using pypdfium2.
    """
    print(f"[*] Rendering PDF with pypdfium2 (DPI={dpi})...")
    pdf = pdfium.PdfDocument(pdf_path)
    scale = dpi / 72.0
    doc_images = []
    doc_texts = []
    for page in pdf:
        doc_images.append(page.render(scale=scale).to_pil())
        text = page.get_textpage().get_text_range().strip()
        doc_texts.append(text if text else "Diagram or mathematical chart page")
    return doc_images, doc_texts

def ingest_pdf(
    pdf_path: str,
    collection_name: str = "scientific_documents",
    remote: bool = False,
    embedder_url: str = "http://localhost:8000",
    embedder_api_key: str = "",
    batch_size: int = 4
):
    if not os.path.exists(pdf_path):
        print(f"[!] PDF file not found at {pdf_path}")
        return

    # 1. MinIO Configuration
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket_name = "scientific-images"

    minio_client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    # 2. Qdrant Configuration
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # 3. Render PDF pages
    doc_images, doc_texts = load_pdf_pages(pdf_path, dpi=150)
    total_pages = len(doc_images)
    print(f"[✓] Rendered {total_pages} pages from {pdf_path}.")

    # 4. Configure Qdrant Collection (ColQwen Multi-vector MaxSim + BM25 Sparse)
    try:
        qdrant_client.get_collection(collection_name)
        print(f"[*] Collection '{collection_name}' already exists.")
    except Exception:
        print(f"[*] Creating collection '{collection_name}' with ColQwen Multi-vector MaxSim & BM25 IDF...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "colqwen": models.VectorParams(
                    size=128,
                    distance=models.Distance.DOT,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )

    # 5. Handle Ingestion: Remote (Vast.ai) or Local
    if remote:
        print(f"[*] Mode: REMOTE GPU Embedder at {embedder_url} (Batch size = {batch_size})...")
        headers = {}
        if embedder_api_key:
            headers["X-API-Key"] = embedder_api_key

        for i in range(0, total_pages, batch_size):
            batch_imgs = doc_images[i:i + batch_size]
            batch_texts = doc_texts[i:i + batch_size]
            current_batch_size = len(batch_imgs)

            # Upload batch images to MinIO and prepare base64 for Embedder
            b64_list = []
            image_urls = []
            for j, img in enumerate(batch_imgs):
                page_idx = i + j + 1
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                
                # MinIO upload
                object_name = f"pdf-images/page_{page_idx}.png"
                img_byte_arr.seek(0)
                minio_client.put_object(
                    bucket_name,
                    object_name,
                    img_byte_arr,
                    length=len(img_bytes),
                    content_type="image/png"
                )
                image_urls.append(f"http://{minio_endpoint}/{bucket_name}/{object_name}")
                b64_list.append(base64.b64encode(img_bytes).decode("utf-8"))

            # Call Remote Embedder API
            try:
                resp = requests.post(
                    f"{embedder_url.rstrip('/')}/embed-images",
                    json={"images_base64": b64_list, "texts": batch_texts},
                    headers=headers,
                    timeout=120
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Embedder API failed ({resp.status_code}): {resp.text}")
                res_data = resp.json()
                colqwen_embs = res_data["colqwen"]
                bm25_embs = res_data["bm25"]
            except Exception as e:
                print(f"[!] Error calling remote embedder for batch {i}-{i+current_batch_size}: {e}")
                continue

            # Upsert batch to Qdrant
            points = []
            for k in range(current_batch_size):
                page_num = i + k + 1
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "colqwen": colqwen_embs[k],
                            "bm25": models.SparseVector(
                                indices=bm25_embs[k]["indices"],
                                values=bm25_embs[k]["values"]
                            )
                        },
                        payload={
                            "document_name": os.path.basename(pdf_path),
                            "page_number": page_num,
                            "page": page_num,
                            "image_url": image_urls[k],
                            "text": batch_texts[k],
                            "total_pages": total_pages
                        }
                    )
                )
            qdrant_client.upsert(collection_name=collection_name, points=points)
            print(f"    -> Successfully indexed pages {i + 1} to {i + current_batch_size}/{total_pages}")

    else:
        print("[*] Mode: LOCAL PyTorch / ColPali Inference...")
        import torch
        from fastembed import SparseTextEmbedding
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        print("[*] Initializing local BM25 Sparse Vector model...")
        bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        bm25_doc_embeddings = list(bm25_model.embed(doc_texts))

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[*] Loading ColQwen2.5 model on {device}...")
        model_id = "vidore/colqwen2.5-v0.2"
        model = ColQwen2_5.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device if device != "mps" else None
        )
        if device == "mps":
            model = model.to("mps")
        model.eval()
        processor = ColQwen2_5_Processor.from_pretrained(model_id)

        print(f"[*] Starting local ingestion of {total_pages} pages...")
        for i in range(total_pages):
            page_num = i + 1
            img = doc_images[i]
            page_text = doc_texts[i]

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            object_name = f"pdf-images/page_{page_num}.png"
            minio_client.put_object(
                bucket_name,
                object_name,
                img_byte_arr,
                length=img_byte_arr.getbuffer().nbytes,
                content_type="image/png"
            )
            image_url = f"http://{minio_endpoint}/{bucket_name}/{object_name}"

            with torch.no_grad():
                if hasattr(processor, "process_images"):
                    batch_input = processor.process_images([img]).to(device)
                else:
                    batch_input = processor(images=[img], return_tensors="pt").to(device)
                emb = model(**batch_input)[0].to("cpu").to(torch.float32).numpy().tolist()

            bm25_sparse = bm25_doc_embeddings[i]

            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "colqwen": emb,
                            "bm25": models.SparseVector(
                                indices=bm25_sparse.indices.tolist(),
                                values=bm25_sparse.values.tolist()
                            )
                        },
                        payload={
                            "document_name": os.path.basename(pdf_path),
                            "page_number": page_num,
                            "page": page_num,
                            "image_url": image_url,
                            "text": page_text,
                            "total_pages": total_pages
                        }
                    )
                ]
            )
            if page_num % 10 == 0 or page_num == total_pages:
                print(f"    -> Successfully indexed page {page_num}/{total_pages}")

    print("[✓] Ingestion completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingest scientific PDF into Qdrant & MinIO")
    parser.add_argument("--pdf", type=str, default="data/Toán rời rạc 2 - 2016.pdf", help="Path to PDF file")
    parser.add_argument("--collection", type=str, default="scientific_documents", help="Qdrant collection name")
    parser.add_argument("--remote", action="store_true", help="Use remote Embedder microservice (e.g. on Vast.ai)")
    parser.add_argument("--embedder-url", type=str, default=os.getenv("EMBEDDER_URL", "http://localhost:8000"), help="URL of Embedder service")
    parser.add_argument("--embedder-api-key", type=str, default=os.getenv("EMBEDDER_API_KEY", ""), help="API key for Embedder service")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for remote embedding")
    args = parser.parse_args()

    ingest_pdf(
        pdf_path=args.pdf,
        collection_name=args.collection,
        remote=args.remote,
        embedder_url=args.embedder_url,
        embedder_api_key=args.embedder_api_key,
        batch_size=args.batch_size
    )

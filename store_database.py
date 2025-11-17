import os
import chromadb
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from typing import List, Dict, Any, Tuple

# Cần cài đặt: pip install chromadb sentence-transformers pyvi

# ==============================================================================
# PHẦN 1: CÁC HÀM LOGIC
# ==============================================================================

def load_embedding_model(model_name: str = 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base') -> SentenceTransformer:
    """
    Chức năng: Tải và trả về model embedding SentenceTransformer.
    """
    print(f"Đang tải model embedding ({model_name})...")
    try:
        model = SentenceTransformer(model_name)
        print("Tải model embedding thành công.")
        return model
    except Exception as e:
        print(f"[LỖI] Không thể tải model embedding: {e}")
        return None

def initialize_chromadb(db_path: str, collection_name: str) -> (chromadb.Collection | None):
    """
    Chức năng: Kết nối với ChromaDB (PersistentClient) và
    trả về collection object.
    """
    print(f"Đang kết nối ChromaDB tại: {db_path}")
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name=collection_name)
        print(f"Đã kết nối collection '{collection_name}'.")
        return collection
    except Exception as e:
        print(f"[LỖI] Không thể kết nối ChromaDB: {e}")
        return None

def process_and_embed_chunks(
    all_chunks: List[Dict[str, Any]], 
    model: SentenceTransformer
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[List[float]]]:
    """
    Chức năng: Nhận danh sách chunks thô (từ file chunker.py),
    thực hiện:
    1. Chuẩn bị 3 list: ids, documents, metadatas.
    2. Tokenize (pyvi) và Tạo embeddings.
    
    Trả về:
        Một tuple chứa 4 danh sách: (ids, documents, metadatas, embeddings)
    """
    ids_list = []
    documents_list = []
    metadatas_list = []
    
    print(f"Đang chuẩn bị {len(all_chunks)} chunks...")
    for chunk in all_chunks:
        ids_list.append(chunk['id'])
        documents_list.append(chunk['text'])
        metadatas_list.append(chunk['metadata'])
    
    print(f"Đang tạo embeddings cho {len(documents_list)} chunks...")
    # Tokenize bằng pyvi trước khi encode (theo logic gốc của bạn)
    tokenized_docs = [tokenize(doc) for doc in documents_list]
    embeddings_list = model.encode(tokenized_docs).tolist()
    print("Đã tạo embeddings thành công!")
    
    return ids_list, documents_list, metadatas_list, embeddings_list

def add_to_collection_in_batches(
    collection: chromadb.Collection,
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
    batch_size: int = 100
):
    """
    Chức năng: Thêm 4 danh sách (ids, docs, metas, embeds) vào
    collection theo từng lô (batch).
    """
    print(f"Đang thêm dữ liệu vào ChromaDB (batch size={batch_size})...")
    total_items = len(ids)
    
    for i in range(0, total_items, batch_size):
        # Lấy batch hiện tại
        batch_ids = ids[i : i + batch_size]
        batch_documents = documents[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        try:
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"   -> Đã thêm batch {i // batch_size + 1}/{total_items // batch_size + 1}")
        except Exception as e:
            print(f"   [LỖI] Thêm batch {i // batch_size + 1} thất bại: {e}")

    print("\n--- HOÀN TẤT ---")
    print(f"Tổng số mục trong collection '{collection.name}': {collection.count()}")

# ==============================================================================
# PHẦN 2: HÀM MAIN ĐIỀU PHỐI
# ==============================================================================

def main_load_database(all_chunks_data: List[Dict[str, Any]]):
    """
    Hàm điều phối chính: Nhận
    `all_chunks_data` (từ file chunker.py) và
    thực hiện toàn bộ quá trình tải database.
    """
    
    # --- 1. CẤU HÌNH ---
    DB_PATH = "/content/drive/MyDrive/Đồ án Deep Learning/my_rag_db_2"
    COLLECTION_NAME = "bai_giang_videos"
    MODEL_NAME = 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base'

    if not all_chunks_data:
        print("[LỖI] Không có dữ liệu 'all_chunks_data' để xử lý. Dừng lại.")
        return

    # --- 2. Tải Model ---
    embedding_model = load_embedding_model(MODEL_NAME)
    if embedding_model is None:
        return

    # --- 3. Kết nối DB ---
    collection = initialize_chromadb(DB_PATH, COLLECTION_NAME)
    if collection is None:
        return

    # --- 4. Tạo Embeddings ---
    ids, docs, metas, embeds = process_and_embed_chunks(
        all_chunks_data, 
        embedding_model
    )

    # --- 5. Thêm vào DB ---
    add_to_collection_in_batches(
        collection=collection,
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeds,
        batch_size=100
    )

# ==============================================================================
# PHẦN 3: ĐIỂM KHỞI CHẠY
# ==============================================================================

if __name__ == "__main__":
    # Đây là cách bạn chạy file này như một script production
    
    # BƯỚC 1: Import file chunker của bạn
    # (Giả sử file trước đó tên là `chunker.py` và nằm cùng thư mục)
    try:
        import chunker 
    except ImportError:
        print("Lỗi: Không tìm thấy file `chunker.py`.")
        print("Hãy đảm bảo `chunker.py` tồn tại để cung cấp dữ liệu.")
        exit(1)

    # BƯỚC 2: Định nghĩa các biến mà `chunker.py` cần
    TRANSCRIPT_DIR = "/content/drive/My Drive/DL_RAG_Video_main/Transcripts_Cleaned/"
    VIDEO_MAP = {
        "video_C1_1_transcript.docx": "https://www.youtube.com/watch?v=6sTFEFKDDqI&t=357s",
        "video_C1_2_transcript.docx": "https://www.youtube.com/watch?v=zuBsXtdWlyQ&t=387s",
        "video_C2_1_transcript.docx": "https://www.youtube.com/watch?v=GdKIVY6CsTw&t=1052s",
        "video_C2_2_1_transcript.docx": "https://www.youtube.com/watch?v=m8uqtMEg8-E&t=762s",
    }

    # BƯỚC 3: Chạy chunker để lấy dữ liệu
    print("--- CHẠY BƯỚC 1: CHUNKING ---")
    all_chunks = chunker.process_directory(VIDEO_MAP, TRANSCRIPT_DIR)

    # BƯỚC 4: Chạy loader database (với dữ liệu vừa chunk)
    print("\n--- CHẠY BƯỚC 2: LOADING DATABASE ---")
    main_load_database(all_chunks)
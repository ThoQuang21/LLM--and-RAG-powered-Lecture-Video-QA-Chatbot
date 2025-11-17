# =======================================================
# PHẦN 0: IMPORTS
# =======================================================
import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from pyvi.ViTokenizer import tokenize
from typing import List, Dict, Any, Optional

# Kiểm tra xem có đang chạy trên Colab không
try:
    from google.colab import drive, userdata
    ON_COLAB = True
except ImportError:
    ON_COLAB = False

# =======================================================
# PHẦN 1: CÁC HÀM TIỆN ÍCH (HELPER FUNCTIONS)
# =======================================================

def parse_timestamp_to_seconds(ts_str: str) -> int:
    """
    Chuyển đổi timestamp string [HH:MM:SS,ms] hoặc [MM:SS,ms]
    sang TỔNG SỐ GIÂY (int).
    """
    if not ts_str or not ts_str.startswith('[') or not ts_str.endswith(']'):
        return 0
    try:
        time_part = ts_str.strip('[]').split(',')[0]
        parts = time_part.split(':')
        total_seconds = 0
        if len(parts) == 3: # HH:MM:SS
            total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2: # MM:SS
            total_seconds = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1: # SS
            total_seconds = int(parts[0])
        return total_seconds
    except Exception as e:
        print(f"Lỗi khi parse timestamp '{ts_str}': {e}")
        return 0

def clean_youtube_url(url: str) -> str:
    """
    Xóa tham số timestamp (&t=...) đã có khỏi URL YouTube.
    """
    if not url:
        return ""
    return url.split('&t=')[0]

# =======================================================
# PHẦN 2: CÁC HÀM KHỞI TẠO (SETUP FUNCTIONS)
# =======================================================

def ket_noi_google_drive(mount_path: str = '/content/drive') -> bool:
    """
    Kết nối và mount Google Drive (chỉ dùng cho Google Colab).
    """
    if ON_COLAB:
        print(f"Đang kết nối với Google Drive tại {mount_path}...")
        try:
            drive.mount(mount_path)
            print("Kết nối Google Drive thành công.")
            return True
        except Exception as e:
            print(f"Lỗi khi kết nối Google Drive: {e}")
            return False
    else:
        print("Không chạy trên Colab, bỏ qua việc mount Drive.")
        return True

def tai_model_embedding(model_name: str) -> Optional[SentenceTransformer]:
    """
    Tải và trả về model SentenceTransformer để tạo embedding.
    """
    print(f"Đang tải model embedding: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("Tải model embedding thành công.")
        return model
    except Exception as e:
        print(f"Lỗi khi tải model embedding: {e}")
        return None

def tai_model_reranker(model_name: str) -> Optional[CrossEncoder]:
    """
    Tải và trả về model CrossEncoder để re-ranking.
    (Dựa trên code mới nhất bạn cung cấp)
    """
    print(f"Đang tải mô hình Re-ranker: {model_name}...")
    try:
        reranker = CrossEncoder(model_name)
        print("Tải mô hình Re-ranker thành công.")
        return reranker
    except Exception as e:
        print(f"Lỗi khi tải Re-ranker: {e}")
        return None

def ket_noi_chromadb(db_path: str, collection_name: str) -> Optional[chromadb.Collection]:
    """
    Kết nối đến Persistent ChromaDB và lấy collection đã tồn tại.
    """
    print(f"Đang kết nối tới ChromaDB tại: {db_path}")
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        print(f"Tải thành công collection '{collection_name}'. Tổng số mục: {collection.count()}")
        return collection
    except Exception as e:
        print(f"Lỗi: Không thể kết nối hoặc tìm thấy collection '{collection_name}'.")
        print(f"Chi tiết lỗi: {e}")
        return None

def thiet_lap_gemini(api_key: str) -> Optional[genai.GenerativeModel]:
    """
    Cấu hình API key cho Gemini và trả về một model có thể sử dụng.
    """
    print("Đang thiết lập Gemini...")
    if not api_key:
        print("Lỗi: Không tìm thấy API Key của Gemini.")
        return None
    try:
        genai.configure(api_key='AIzaSyBD27hwT7Zu1yACDlbR1sEoVDKww2T2Cuo')
        model = genai.GenerativeModel('gemini-2.5-flash') 
        print("Thiết lập Gemini thành công.")
        return model
    except Exception as e:
        print(f"Lỗi khi cấu hình Gemini: {e}")
        return None

# =======================================================
# PHẦN 3: HÀM PIPELINE RAG CHÍNH (GIAI ĐOẠN 1-9)
# =======================================================

def get_rag_answer_pipeline(
    original_query_text: str,
    llm_model: genai.GenerativeModel,
    embedding_model: SentenceTransformer,
    collection: chromadb.Collection,
    reranker: CrossEncoder,
    # Cấu hình có thể điều chỉnh
    per_query_k: int = 7,
    score_threshold: float = 0.0,
    min_docs_needed: int = 2,
    fallback_k: int = 3
) -> Dict[str, Optional[str]]:
    """
    Thực thi toàn bộ pipeline RAG phức tạp (Giai đoạn 1-9).
    """
    
    print(f"\n==============================================")
    print(f"🚀 BẮT ĐẦU PIPELINE RAG CHO TRUY VẤN: '{original_query_text}'")
    print(f"==============================================")

    # --- GIAI ĐOẠN 2: BIẾN ĐỔI TRUY VẤN (HyDE + Multi-Query) ---
    print("\n--- Giai đoạn 2: Đang biến đổi truy vấn... ---")
    
    # 2A. MULTI-QUERY
    transform_prompt = f"""
Bạn là một chuyên gia phân tích truy vấn.
Hãy đọc câu hỏi của người dùng và phân rã nó thành 3 câu hỏi con, mỗi câu hỏi khai thác một khía cạnh khác nhau (định nghĩa, bản chất, mục đích).
**Yêu cầu:** Chỉ trả lời bằng các câu hỏi con, mỗi câu hỏi trên một dòng.
**Câu hỏi gốc:** "{original_query_text}"
**Các câu hỏi con (phân rã):**
"""
    transform_config = genai.types.GenerationConfig(temperature=0.0)
    generated_queries = []
    try:
        transform_response = llm_model.generate_content(transform_prompt, generation_config=transform_config)
        sub_queries_text = transform_response.text
        generated_queries = [q.strip() for q in sub_queries_text.split('\n') if q.strip()]
    except Exception as e:
        print(f"Lỗi khi tạo sub-query: {e}")

    # 2B. HYDE
    print("Đang tạo tài liệu giả lập (HyDE)...")
    hyde_prompt = f"""
Hãy viết một đoạn văn ngắn (khoảng 2-3 câu) trả lời trực tiếp cho câu hỏi sau.
Đoạn văn này sẽ được dùng để tìm kiếm các tài liệu tương tự.
Hãy tập trung vào các từ khóa và khái niệm cốt lõi.
**Câu hỏi:** "{original_query_text}"
**Câu trả lời giả lập:**
"""
    hyde_config = genai.types.GenerationConfig(temperature=0.3)
    hyde_document_text = ""
    try:
        hyde_response = llm_model.generate_content(hyde_prompt, generation_config=hyde_config)
        hyde_document_text = hyde_response.text.strip().replace("\n", " ")
    except Exception as e:
        print(f"Lỗi khi tạo HyDE doc: {e}")

    # 2C. TỔNG HỢP
    all_search_texts = [original_query_text] + generated_queries
    if hyde_document_text:
        all_search_texts.append(hyde_document_text)
    print(f"Đã tạo {len(all_search_texts)} văn bản để tìm kiếm (gốc + con + HyDE).")


    # --- GIAI ĐOẠN 3: RETRIEVAL ---
    all_retrieved_docs_map = {}
    print(f"\n--- Giai đoạn 3: Đang truy xuất cho {len(all_search_texts)} văn bản (k={per_query_k} mỗi văn bản)... ---")

    for search_text in all_search_texts:
        try:
            search_tokenized = tokenize(search_text)
            search_embedding = embedding_model.encode([search_tokenized])
            results = collection.query(
                query_embeddings=search_embedding.tolist(),
                n_results=per_query_k,
                include=["metadatas", "documents"]
            )
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            for doc_text, meta in zip(documents, metadatas):
                if doc_text not in all_retrieved_docs_map:
                    all_retrieved_docs_map[doc_text] = meta
        except Exception as e:
            print(f"Lỗi khi truy xuất cho văn bản '{search_text[:50]}...': {e}")

    retrieved_doc_texts = list(all_retrieved_docs_map.keys())
    if not retrieved_doc_texts:
        print("Lỗi: Không truy xuất được bất kỳ tài liệu nào. Dừng pipeline.")
        return {"answer": "Xin lỗi, tôi không tìm thấy tài liệu nào liên quan đến câu hỏi này.", "source_url": None}
    print(f"Đã truy xuất được tổng cộng {len(retrieved_doc_texts)} chunks (duy nhất).")


    # --- GIAI ĐOẠN 4: RE-RANKING ---
    print(f"\n--- Giai đoạn 4: Đang Re-ranking {len(retrieved_doc_texts)} chunks... ---")
    query_chunk_pairs = [[original_query_text, doc_text] for doc_text in retrieved_doc_texts]
    try:
        scores = reranker.predict(query_chunk_pairs)
        retrieved_metadatas = [all_retrieved_docs_map[text] for text in retrieved_doc_texts]
        scored_chunks_with_meta = sorted(
            list(zip(scores, retrieved_doc_texts, retrieved_metadatas)),
            key=lambda x: x[0],
            reverse=True
        )
        print("Đã chấm điểm và sắp xếp xong.")
    except Exception as e:
        print(f"Lỗi khi re-ranking: {e}")
        scored_chunks_with_meta = [(0.0, text, all_retrieved_docs_map[text]) for text in retrieved_doc_texts]


    # --- GIAI ĐOẠN 4.5: TRÍCH XUẤT LINK TỐT NHẤT ---
    best_source_url = None
    if scored_chunks_with_meta:
        top_score, top_text, top_meta = scored_chunks_with_meta[0]
        if top_meta and 'source_url' in top_meta and 'start_time' in top_meta:
            original_url = top_meta['source_url']
            start_time_str = top_meta['start_time']
            base_url = clean_youtube_url(original_url)
            total_seconds = parse_timestamp_to_seconds(start_time_str)
            if total_seconds > 0:
                best_source_url = f"{base_url}&t={total_seconds}s"
            else:
                best_source_url = base_url
            print(f"Đã trích xuất nguồn tốt nhất: {best_source_url}")
        elif top_meta and 'source_url' in top_meta:
            best_source_url = top_meta['source_url']
            print(f"Đã trích xuất nguồn (không có start_time): {best_source_url}")
        else:
            print("Không tìm thấy 'source_url' trong metadata của Hạng 1.")
    else:
        print("Không có chunk nào để trích xuất link.")


    # --- GIAI ĐOẠN 4d: LỌC ---
    threshold_chunks = [item for item in scored_chunks_with_meta if item[0] >= score_threshold]
    if len(threshold_chunks) < min_docs_needed:
        print(f"Chỉ tìm thấy {len(threshold_chunks)} chunk vượt ngưỡng. Fallback về Top-{fallback_k}.")
        final_best_chunks = scored_chunks_with_meta[:fallback_k]
    else:
        print(f"Đã tìm thấy {len(threshold_chunks)} chunk vượt ngưỡng (đủ yêu cầu).")
        final_best_chunks = threshold_chunks
    print(f"Tổng cộng sẽ dùng {len(final_best_chunks)} chunk (đã lọc) để lấy context.")


    # --- GIAI ĐOẠN 5: Xây dựng Context ---
    final_context_windows_texts = []
    for score, text, meta in final_best_chunks:
        if meta and 'context_window' in meta:
            final_context_windows_texts.append(meta['context_window'])
        else:
            final_context_windows_texts.append(text)
    unique_contexts = list(dict.fromkeys(final_context_windows_texts))
    context_string = "\n\n".join(unique_contexts)


    # --- GIAI ĐOẠN 6: PROMPT ---
    strict_prompt = f"""
Bạn là một trợ lý AI, nhiệm vụ của bạn là trả lời câu hỏi CHỈ DỰA TRÊN bối cảnh (context) được cung cấp.
Hãy đọc kỹ đoạn văn bản dưới đây:
--- (BẮT ĐẦU VĂN BẢN) ---
{context_string}
--- (KẾT THÚC VĂN BẢN) ---
Dựa trên văn bản trên, hãy trả lời câu hỏi sau:
Câu hỏi: {original_query_text}
**Yêu cầu nghiêm ngặt:**
1. Đọc kỹ bối cảnh đã cho.
2. Câu trả lời của bạn PHẢI được rút ra trực tiếp từ thông tin có trong bối cảnh.
3. **QUAN TRỌNG:** Nếu thông tin để trả lời câu hỏi không có trong bối cảnh, hãy trả lời chính xác một câu: "Tôi không tìm thấy thông tin này trong bối cảnh được cung cấp."
4. Không được suy diễn, không được thêm kiến thức bên ngoài, không được "chém gió".
**Câu trả lời (dựa trên bối cảnh):**
"""

    # --- GIAI ĐOẠN 7: Cấu hình Generation ---
    config = genai.types.GenerationConfig(temperature=0.1)
    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    # --- GIAI ĐOẠN 8 & 9: GENERATE và Trả về kết quả ---
    print("\n--- Giai đoạn 8: Đang gửi prompt nghiêm ngặt đến Gemini... ---")
    final_answer = ""
    try:
        response = llm_model.generate_content(
            strict_prompt,
            generation_config=config,
            safety_settings=safety_settings
        )
        final_answer = response.text.strip()
        print("Đã nhận được câu trả lời.")
    except Exception as e:
        print(f"Xảy ra lỗi khi gọi Gemini: {e}")
        final_answer = f"Xin lỗi, đã xảy ra lỗi trong quá trình tạo câu trả lời: {e}"

    print("🏁 PIPELINE RAG HOÀN TẤT.")
    
    return {
        "answer": final_answer,
        "source_url": best_source_url
    }

# =======================================================
# PHẦN 4: HÀM MAIN THỰC THI
# =======================================================

def main():
    """
    Hàm thực thi chính:
    1. Thiết lập hằng số (đường dẫn, tên model).
    2. Lấy API Key.
    3. Gọi các hàm trong PHẦN 2 để khởi tạo tất cả thành phần.
    4. Kiểm tra lỗi.
    5. Đặt câu hỏi và gọi hàm pipeline (PHẦN 3).
    6. In kết quả cuối cùng.
    """
    
    # --- 1. Thiết lập hằng số ---
    print("--- ⚙️ BẮT ĐẦU KHỞI CHẠY RAG PIPELINE ⚙️ ---")
    DB_PATH = "/content/drive/MyDrive/DL_RAG_Video_main/my_rag_db_2"
    COLLECTION_NAME = "bai_giang_videos"
    EMBEDDING_MODEL_NAME = 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base'
    RERANKER_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
    DRIVE_MOUNT_PATH = '/content/drive'

    # --- 2. Lấy API Key (An toàn) ---
    GEMINI_API_KEY = 'AIzaSyBD27hwT7Zu1yACDlbR1sEoVDKww2T2Cuo'
    if ON_COLAB:
        try:
            # Lấy key từ Colab Secrets (biểu tượng chìa khóa 🔑)
            GEMINI_API_KEY = userdata.get('GOOGLE_API_KEY') 
        except Exception as e:
            print(f"Không thể lấy 'GOOGLE_API_KEY' từ Colab Secrets: {e}")
    else:
        # Lấy key từ biến môi trường nếu chạy local
        GEMINI_API_KEY = os.environ.get('GOOGLE_API_KEY')

    if not GEMINI_API_KEY:
        print("\n❌ LỖI: Không tìm thấy 'GOOGLE_API_KEY'.")
        print("Vui lòng thiết lập biến này trong Colab Secrets hoặc môi trường của bạn.")
        return

    # --- 3. Khởi tạo tất cả thành phần ---
    
    # Kết nối Drive (cần thiết nếu DB_PATH ở trên Drive)
    if ON_COLAB:
        ket_noi_google_drive(DRIVE_MOUNT_PATH)
    
    # Tải các model và kết nối DB
    # (Chúng ta gọi tuần tự để log dễ đọc hơn)
    model_embed = tai_model_embedding(EMBEDDING_MODEL_NAME)
    model_rerank = tai_model_reranker(RERANKER_MODEL_NAME)
    rag_collection = ket_noi_chromadb(DB_PATH, COLLECTION_NAME)
    llm_model = thiet_lap_gemini(GEMINI_API_KEY)

    # --- 4. Kiểm tra ---
    all_components = {
        "Embedding Model": model_embed,
        "Reranker Model": model_rerank,
        "ChromaDB Collection": rag_collection,
        "LLM Model": llm_model
    }
    
    if not all(all_components.values()):
        print("\n❌ LỖI: Một hoặc nhiều thành phần không thể khởi tạo.")
        for name, component in all_components.items():
            if not component:
                print(f"    - {name}: KHỞI TẠO THẤT BẠI")
        print("Vui lòng kiểm tra lại lỗi bên trên và đường dẫn. Dừng chương trình.")
        return
        
    print("\n===================================")
    print("✅ TẤT CẢ MODEL VÀ DB ĐÃ SẴN SÀNG!")
    print("===================================")

    # --- 5. Đặt câu hỏi và chạy pipeline ---
    
    # ▼▼▼ ĐÂY LÀ NƠI BẠN ĐẶT CÂU HỎI CỦA MÌNH ▼▼▼
    query = "Attention khác Self-Attention ở điểm nào"
    
    result = get_rag_answer_pipeline(
        original_query_text=query,
        llm_model=llm_model,
        embedding_model=model_embed,
        collection=rag_collection,
        reranker=model_rerank
    )

    # --- 6. In kết quả cuối cùng ---
    print("\n\n================ KẾT QUẢ CUỐI CÙNG ================")
    print(f"❓ HỎI: {query}\n")
    print(f"🤖 TRẢ LỜI:\n{result.get('answer', 'Không có câu trả lời')}\n")
    print(f"🔗 NGUỒN THAM KHẢO:\n{result.get('source_url', 'Không có nguồn')}")
    print("==================================================")

# =======================================================
# PHẦN 5: ĐIỂM BẮT ĐẦU CHẠY CODE
# =======================================================
if __name__ == "__main__":
    main()
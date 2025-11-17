# =======================================================
# STREAMLIT APP - HỆ THỐNG HỎI ĐÁP VỚI RAG
# =======================================================
import streamlit as st
import sys
import os
from optimize_latency import (
    tai_model_embedding,
    tai_model_reranker,
    ket_noi_chromadb,
    thiet_lap_gemini,
    get_rag_answer_pipeline
)

# Thông điệp fallback khi không tìm thấy câu trả lời trong bối cảnh
FALLBACK_MESSAGE = "Tôi không tìm thấy thông tin này trong bối cảnh được cung cấp."

# =======================================================
# CẤU HÌNH TRANG
# =======================================================
st.set_page_config(
    page_title="🤖 Hệ Thống Hỏi Đáp RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================================================
# KHỞI TẠO MODELS (CACHED)
# =======================================================
@st.cache_resource
def khoi_tao_models():
    """
    Khởi tạo tất cả các models và components một lần duy nhất.
    Sử dụng cache để tránh reload mỗi lần người dùng đặt câu hỏi.
    """
    # Cấu hình
    DB_PATH = "my_rag_db_2"
    COLLECTION_NAME = "bai_giang_videos"
    EMBEDDING_MODEL_NAME = 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base'
    RERANKER_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
    GEMINI_API_KEY = 'AIzaSyAqJn039l1ThNaNATJ_4wTIgHv0hrxKRWE'
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Khởi tạo từng component
    status_text.text("🔄 Đang tải Embedding Model...")
    progress_bar.progress(10)
    model_embed = tai_model_embedding(EMBEDDING_MODEL_NAME)
    
    status_text.text("🔄 Đang tải Reranker Model...")
    progress_bar.progress(30)
    model_rerank = tai_model_reranker(RERANKER_MODEL_NAME)
    
    status_text.text("🔄 Đang kết nối ChromaDB...")
    progress_bar.progress(50)
    rag_collection = ket_noi_chromadb(DB_PATH, COLLECTION_NAME)
    
    status_text.text("🔄 Đang thiết lập Gemini...")
    progress_bar.progress(70)
    llm_model = thiet_lap_gemini(GEMINI_API_KEY)
    
    progress_bar.progress(100)
    status_text.text("✅ Đã tải xong tất cả models!")
    
    # Kiểm tra lỗi
    if not all([model_embed, model_rerank, rag_collection, llm_model]):
        st.error("❌ Lỗi: Không thể khởi tạo một hoặc nhiều components!")
        return None
    
    return {
        "embedding_model": model_embed,
        "reranker_model": model_rerank,
        "collection": rag_collection,
        "llm_model": llm_model
    }

# =======================================================
# HÀM XỬ LÝ CÂU HỎI
# =======================================================
def xu_ly_cau_hoi(query, models, per_query_k=5, max_docs_to_rerank=20, rerank_batch_size=32):
    """
    Xử lý câu hỏi và trả về kết quả từ pipeline RAG.
    """
    try:
        # Tạo placeholder cho progress
        progress_placeholder = st.empty()
        
        result = get_rag_answer_pipeline(
            original_query_text=query,
            llm_model=models["llm_model"],
            embedding_model=models["embedding_model"],
            collection=models["collection"],
            reranker=models["reranker_model"],
            per_query_k=per_query_k,
            max_docs_to_rerank=max_docs_to_rerank,
            rerank_batch_size=rerank_batch_size
        )
        
        # Xóa placeholder sau khi hoàn thành
        progress_placeholder.empty()
        return result
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý câu hỏi: {e}")
        import traceback
        with st.expander("Chi tiết lỗi"):
            st.code(traceback.format_exc())
        return None

# =======================================================
# GIAO DIỆN CHÍNH
# =======================================================
def main():
    # Header
    st.title("🤖 Hệ Thống Hỏi Đáp RAG")
    st.markdown("---")
    st.markdown("""
    Hệ thống hỏi đáp thông minh sử dụng RAG (Retrieval-Augmented Generation).
    Nhập câu hỏi của bạn và nhận câu trả lời kèm link YouTube tham khảo.
    """)
    
    # Sidebar cho cấu hình
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        st.markdown("---")
        
        per_query_k = st.slider("Số lượng docs mỗi query (per_query_k)", 3, 10, 5)
        max_docs_to_rerank = st.slider("Số lượng docs để re-rank", 10, 30, 20)
        rerank_batch_size = st.slider("Batch size cho re-ranker", 16, 64, 32)
        
        st.markdown("---")
        st.markdown("### 💡 Gợi ý")
        st.info("""
        - Giảm `per_query_k` để tăng tốc độ
        - Giảm `max_docs_to_rerank` để giảm thời gian re-ranking
        - Tăng batch size nếu có GPU mạnh
        """)
    
    # Khởi tạo models
    if 'models_loaded' not in st.session_state:
        with st.container():
            st.info("🔄 Đang khởi tạo models lần đầu tiên... Vui lòng đợi trong giây lát...")
            models = khoi_tao_models()
            if models:
                st.session_state.models_loaded = True
                st.session_state.models = models
                st.success("✅ Đã khởi tạo thành công! Bạn có thể đặt câu hỏi.")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Không thể khởi tạo models. Vui lòng kiểm tra lại cấu hình.")
                st.stop()
    else:
        models = st.session_state.models
        st.success("✅ Models đã sẵn sàng!")
    
    st.markdown("---")
    
    # Ô nhập câu hỏi
    st.subheader("📝 Nhập câu hỏi của bạn")
    query = st.text_area(
        "Câu hỏi:",
        placeholder="Ví dụ: Attention khác Self-Attention ở điểm nào?",
        height=100,
        key="query_input"
    )
    
    # Nút gửi
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit_button = st.button("🚀 Gửi câu hỏi", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Xóa", use_container_width=True)
    
    if clear_button:
        st.session_state.query_input = ""
        st.rerun()
    
    # Xử lý khi người dùng gửi câu hỏi
    if submit_button and query:
        # Hiển thị thông báo đang xử lý
        status_container = st.container()
        with status_container:
            st.info(f"🔄 Đang xử lý câu hỏi: **{query}**\n\nVui lòng đợi trong giây lát...")
        
        result = xu_ly_cau_hoi(
            query, 
            models,
            per_query_k=per_query_k,
            max_docs_to_rerank=max_docs_to_rerank,
            rerank_batch_size=rerank_batch_size
        )
        
        # Xóa thông báo đang xử lý
        status_container.empty()
        
        if result:
            st.markdown("---")
            st.subheader("💬 Kết quả")
            
            # Hiển thị câu hỏi và câu trả lời
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**❓ Câu hỏi:**")
            with col2:
                st.markdown(f"*{query}*")
            
            answer = result.get("answer", "Không có câu trả lời")
            st.markdown("**🤖 Trả lời:**")
            st.success(answer)
            
            answer_is_fallback = answer.strip() == FALLBACK_MESSAGE
            
            # Hiển thị link YouTube (nếu có) và câu trả lời không phải fallback
            source_url = result.get("source_url")
            if source_url and not answer_is_fallback:
                st.markdown("---")
                st.subheader("🔗 Nguồn tham khảo")
                
                # Kiểm tra xem có phải link YouTube không
                if "youtube.com" in source_url or "youtu.be" in source_url:
                    # Lấy video ID để embed
                    video_id = None
                    if "watch?v=" in source_url:
                        video_id = source_url.split("watch?v=")[1].split("&")[0]
                    elif "youtu.be/" in source_url:
                        video_id = source_url.split("youtu.be/")[1].split("?")[0]
                    
                    if video_id:
                        # Hiển thị video embed
                        st.video(f"https://www.youtube.com/watch?v={video_id}")
                    
                    # Hiển thị link có thể click
                    st.markdown("**📺 Link YouTube:**")
                    st.markdown(
                        f'<a href="{source_url}" target="_blank" style="font-size: 16px; color: #FF0000; text-decoration: none;">🔗 {source_url}</a>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("**🔗 Link tham khảo:**")
                    st.markdown(
                        f'<a href="{source_url}" target="_blank" style="font-size: 16px; text-decoration: none;">🔗 {source_url}</a>',
                        unsafe_allow_html=True
                    )
            elif answer_is_fallback:
                st.info("Nguồn tham khảo không được cung cấp khi không tìm thấy thông tin trong bối cảnh.")
            else:
                st.warning("⚠️ Không tìm thấy nguồn tham khảo.")
            
            st.markdown("---")
    
    elif submit_button and not query:
        st.warning("⚠️ Vui lòng nhập câu hỏi của bạn!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Hệ thống RAG với Streamlit | Powered by Gemini, ChromaDB, Sentence Transformers</p>
    </div>
    """, unsafe_allow_html=True)

# =======================================================
# CHẠY APP
# =======================================================
if __name__ == "__main__":
    main()


import os
import uuid
import re
from docx import Document
from typing import List, Dict, Any, Tuple, Optional

# Cần cài đặt: pip install python-docx nltk

# --- TÙY CHỈNH CẤU HÌNH ---
CHILD_CHUNK_MAX_LENGTH = 600
OVERLAP_SENTENCES = 2
PARENT_CONTEXT_BUFFER = 4

# ==============================================================================
# PHẦN 1: HÀM HỖ TRỢ NLTK (Tokenizer)
# ==============================================================================

def _safe_sent_tokenize(text: str) -> List[str]:
    """
    Hàm nội bộ để tách câu (Dùng NLTK cho logic phức tạp).
    """
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("tokenizers/punkt_tab") # Resource tiếng Việt
        except LookupError:
            print("Đang tải NLTK 'punkt' và 'punkt_tab' (lần đầu)...")
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True) # Tải resource tiếng Việt

        try:
            return nltk.sent_tokenize(text, language="vietnamese")
        except Exception:
            return nltk.sent_tokenize(text)

    except ImportError:
        print("[LỖI] Logic chunking phức tạp yêu cầu NLTK.")
        print("Vui lòng chạy: pip install nltk")
        pattern = r'(?<=[\.\!\?\u2026])\s+'
        pieces = re.split(pattern, text.strip())
        return [p.strip() for p in pieces if p and not p.isspace()]

# ==============================================================================
# PHẦN 2: HÀM ĐỌC VÀ TRÍCH XUẤT (PARSE)
# ==============================================================================

def read_cleaned_docx_with_timestamps(file_path: str) -> List[Tuple[str, str]]:
    """
    Chức năng: Đọc file .docx đã sửa (SỬA LỖI 'SHIFT+ENTER').
    (Logic này từ tin nhắn #69)
    """
    structured_data = []
    try:
        doc = Document(file_path)

        full_text_from_all_paras = "\n".join(
            para.text for para in doc.paragraphs if para.text.strip()
        )

        lines = re.split(r'[\n\r]+', full_text_from_all_paras)

        if not lines:
            print(f"Lỗi: Không đọc được nội dung từ {file_path}")
            return []

        print(f"  Đã đọc {len(lines)} dòng/đoạn (từ soft-break) trong file.")

        for line in lines:
            text = line.strip()
            if not text:
                continue

            start_bracket = text.find('[')
            end_bracket = text.find(']')

            if start_bracket != -1 and end_bracket != -1 and start_bracket < 5:
                timestamp_str = text[start_bracket : end_bracket + 1].strip()
                paragraph_text = text[end_bracket + 1 :].strip()

                if timestamp_str.startswith('[') and ':' in timestamp_str:
                    structured_data.append((timestamp_str, paragraph_text))
                else:
                    print(f"  [CẢNH BÁO] Dòng (dùng find) không hợp lệ (gán 00:00): {text[:50]}...")
                    structured_data.append(("[00:00,000]", text))
            else:
                print(f"  [CẢNH BÁO] Dòng không tìm thấy [timestamp] ở đầu (gán 00:00): {text[:50]}...")
                structured_data.append(("[00:00,000]", text))

        return structured_data
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return []

# ==============================================================================
# PHẦN 3: HÀM CHUNKING CHÍNH (LOGIC TOÀN CỤC)
# ==============================================================================

def chunk_data_globally(
    structured_data: List[Tuple[str, str]],
    video_url: str
) -> List[Dict[str, Any]]:
    """
    Chức năng: Dùng logic chunking "phức tạp" (Toàn cục).
    (Logic này từ tin nhắn #65 - đã sửa lỗi .strip())
    """

    all_sentences = []
    sentence_to_timestamp_map = {} # Map để tra cứu timestamp cho mỗi câu

    # 1. Tiền xử lý: Tách TẤT CẢ câu và map chúng với timestamp
    print("  Đang tách câu (toàn cục) và map timestamp...")
    for timestamp_str, para_text in structured_data:
        sentences_in_para = _safe_sent_tokenize(para_text)
        all_sentences.extend(sentences_in_para)

        # Gán TẤT CẢ câu trong đoạn này cho timestamp của đoạn đó
        for sent in sentences_in_para:
            if sent not in sentence_to_timestamp_map:
                 sentence_to_timestamp_map[sent] = timestamp_str

    # 2. Bắt đầu chunking trên danh sách TOÀN CỤC (all_sentences)
    chunks_list = []
    i = 0
    n = len(all_sentences)
    print(f"  Đang chunking phức tạp (toàn cục) {n} câu...")

    while i < n:
        current_chunk_words = [] # Sẽ chứa các câu GỐC (chưa strip)
        current_len = 0
        temp_index = i

        while temp_index < n:
            sent = all_sentences[temp_index] # Lấy câu GỐC

            if not sent.strip():
                temp_index += 1
                continue

            # --- XỬ LÝ CÂU QUÁ DÀI ---
            if len(sent.strip()) > CHILD_CHUNK_MAX_LENGTH:
                # 1A. Flush chunk "con" TRƯỚC ĐÓ (nếu có)
                if current_len > 0:
                    chunk_text = " ".join(current_chunk_words).strip()
                    if chunk_text:
                        parent_start_idx = max(0, i - PARENT_CONTEXT_BUFFER)
                        parent_end_idx = min(n, (temp_index - 1) + PARENT_CONTEXT_BUFFER + 1)
                        context_window_text = " ".join(all_sentences[parent_start_idx:parent_end_idx]).strip()

                        first_sent = current_chunk_words[0]
                        chunk_start_time = sentence_to_timestamp_map.get(first_sent, "[00:00,000]")

                        chunks_list.append({
                            "id": str(uuid.uuid4()),
                            "text": chunk_text,
                            "metadata": {
                                "source_url": video_url,
                                "start_time": chunk_start_time,
                                "context_window": context_window_text
                            }
                        })
                    current_chunk_words = []
                    current_len = 0

                # 1B. Xử lý câu "con" quá dài
                parent_start_idx = max(0, temp_index - PARENT_CONTEXT_BUFFER)
                parent_end_idx = min(n, temp_index + PARENT_CONTEXT_BUFFER + 1)
                context_window_text = " ".join(all_sentences[parent_start_idx:parent_end_idx]).strip()

                chunk_start_time = sentence_to_timestamp_map.get(sent, "[00:00,000]")

                words = sent.strip().split()
                w_buf = []
                w_len = 0
                for w in words:
                    add_len = len(w) if w_len == 0 else (1 + len(w))
                    if w_len + add_len > CHILD_CHUNK_MAX_LENGTH:
                        sub_chunk = " ".join(w_buf).strip()
                        if sub_chunk:
                            chunks_list.append({
                                "id": str(uuid.uuid4()),
                                "text": sub_chunk,
                                "metadata": {
                                    "source_url": video_url,
                                    "start_time": chunk_start_time,
                                    "context_window": context_window_text
                                }
                            })
                        w_buf = [w]
                        w_len = len(w)
                    else:
                        w_buf.append(w)
                        w_len += add_len

                if w_buf:
                    chunks_list.append({
                        "id": str(uuid.uuid4()),
                        "text": " ".join(w_buf).strip(),
                        "metadata": {
                            "source_url": video_url,
                            "start_time": chunk_start_time,
                            "context_window": context_window_text
                        }
                    })

                temp_index += 1
                i = temp_index
                break

            # --- XỬ LÝ CÂU BÌNH THƯỜNG ---
            else:
                sent_len = len(sent.strip())
                add_len = sent_len if current_len == 0 else (1 + sent_len)
                if current_len + add_len > CHILD_CHUNK_MAX_LENGTH:
                    break

                current_chunk_words.append(sent) # Append câu GỐC
                current_len += add_len
                temp_index += 1

        # --- XỬ LÝ FLUSH CHUNK "CON" BÌNH THƯỜNG ---
        if current_len > 0 and current_chunk_words:
            chunk_text = " ".join(current_chunk_words).strip()

            parent_start_idx = max(0, i - PARENT_CONTEXT_BUFFER)
            parent_end_idx = min(n, (temp_index - 1) + PARENT_CONTEXT_BUFFER + 1)
            context_window_text = " ".join(all_sentences[parent_start_idx:parent_end_idx]).strip()

            first_sent = current_chunk_words[0]
            chunk_start_time = sentence_to_timestamp_map.get(first_sent, "[00:00,000]")

            chunks_list.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": {
                    "source_url": video_url,
                    "start_time": chunk_start_time,
                    "context_window": context_window_text
                }
            })

        if temp_index >= n:
            break

        new_i = max(i + 1, temp_index - OVERLAP_SENTENCES)
        if new_i <= i:
            new_i = i + 1
        i = new_i

    return chunks_list

# ==============================================================================
# PHẦN 4: HÀM ĐIỀU PHỐI (MAIN SCRIPT)
# ==============================================================================

def process_directory(
    video_map: Dict[str, str],
    transcript_dir: str
) -> List[Dict[str, Any]]:
    """
    Hàm chính để chạy toàn bộ quy trình:
    Lặp qua video_map, đọc file, chunk, và gộp kết quả.
    """
    all_chunks_data = []
    print("Bắt đầu xử lý và chunking (Phức tạp + Toàn cục) các file...")

    for file_name, video_url in video_map.items():
        file_path = os.path.join(transcript_dir, file_name)

        if not os.path.exists(file_path):
            print(f"[BỎ QUA] Không tìm thấy file: {file_name}")
            continue

        # 1. Đọc nội dung (Đã sửa lỗi Shift+Enter)
        print(f"\nĐang đọc: {file_name}")
        structured_data = read_cleaned_docx_with_timestamps(file_path)

        if structured_data:
            # 2. Chunk (Hàm Phức tạp + Toàn cục)
            chunks = chunk_data_globally(structured_data, video_url)

            # 3. Thêm vào danh sách tổng
            all_chunks_data.extend(chunks)
            print(f"  -> Đã chunk thành {len(chunks)} phần.")

    print("\n--- HOÀN TẤT ---")
    print(f"Tổng cộng đã tạo được {len(all_chunks_data)} chunks từ {len(video_map)} video.")
    return all_chunks_data

# ==============================================================================
# PHẦN 5: ĐIỂM KHỞI CHẠY (MAIN)
# ==============================================================================



if __name__ == "__main__":

    # --- ĐỊNH NGHĨA BIẾN ---

    # Thư mục chứa các file .docx đã được Gemini sửa sạch
    TRANSCRIPT_DIR = "/content/drive/My Drive/DL_RAG_Video_main/Transcripts_Cleaned/"

    # Map (lấy từ Script 1)
    VIDEO_MAP = {
    "video_C1_1_transcript.docx": "https://www.youtube.com/watch?v=6sTFEFKDDqI&t=357s",
    "video_C1_2_transcript.docx": "https://www.youtube.com/watch?v=zuBsXtdWlyQ&t=387s",
    "video_C2_1_transcript.docx": "https://www.youtube.com/watch?v=GdKIVY6CsTw&t=1052s",
    "video_C2_2_1_transcript.docx": "https://www.youtube.com/watch?v=m8uqtMEg8-E&t=762s",
    "video_C2_2_2_transcript.docx": "https://www.youtube.com/watch?v=MtJDVr5xHB4&t=785s",
    "video_C2_3_transcript.docx": "https://www.youtube.com/watch?v=T2xJmTiRM5o&t=608s",
    "video_C2_4_transcript.docx": "https://www.youtube.com/watch?v=G4lcEPrfETo&t=13s",
    "video_C2_5_transcript.docx": "https://www.youtube.com/watch?v=aXB_C9IAyMg&t=458s",
    "video_C3_1_transcript.docx": "https://www.youtube.com/watch?v=q3oZyk3l8EU&t=718s",
    "video_C3_2_1_transcript.docx": "https://www.youtube.com/watch?v=SKcHedTJIL0&t=898s",
    "video_C3_2_2_transcript.docx": "https://www.youtube.com/watch?v=A3iFEk5jllM&t=600s",
    "video_C3_3_1_transcript.docx": "https://www.youtube.com/watch?v=KeNRQw9j_ps&t=1003s",
    "video_C3_3_2_transcript.docx": "https://www.youtube.com/watch?v=TNrJYPuDADM&t=642s",
    "video_C3_3_3_transcript.docx": "https://www.youtube.com/watch?v=rVpEwMijtvQ&t=421s",
    "video_C3_4_1_transcript.docx": "https://www.youtube.com/watch?v=7YLMIKqygPU&t=6s",
    "video_C3_4_2_transcript.docx": "https://www.youtube.com/watch?v=gmQTGRTHH2o&t=589s",
    "video_C4_1_1_transcript.docx": "https://www.youtube.com/watch?v=PyC3pl_r8jw&t=753s",
    "video_C4_1_2_transcript.docx": "https://www.youtube.com/watch?v=KoBIBuqGb9A&t=531s",
    "video_C4_1_3_transcript.docx": "https://www.youtube.com/watch?v=tMKUb4k5nZw&t=499s",
    "video_C4_1_4_transcript.docx": "https://www.youtube.com/watch?v=MNHY9TA4fZs&t=322s",
    "video_C4_2_transcript.docx": "https://www.youtube.com/watch?v=0I8uw0ELYj4&t=504s",
    "video_C5_1_2_transcript.docx": "https://www.youtube.com/watch?v=RVFApjx4KKI&t=111s",
    "video_C5_3_transcript.docx": "https://www.youtube.com/watch?v=Til9AdPO7JE",
    "video_C5_4_transcript.docx": "https://www.youtube.com/watch?v=4p0L74qD7Lg",
    "video_C6_1_transcript.docx": "https://www.youtube.com/watch?v=30kCjQ0BdUc&t=915s",
    "video_C6_2_transcript.docx": "https://www.youtube.com/watch?v=utOha-d0prc&t=2s",
    "video_C6_3_transcript.docx": "https://www.youtube.com/watch?v=O57P9YHZOE0&t=573s",
    "video_C6_4_1_transcript.docx": "https://www.youtube.com/watch?v=UJNyIptbcNM&t=825s",
    "video_C6_4_2_transcript.docx": "https://www.youtube.com/watch?v=AkHEcgasvkw&t=335s",
    "video_C6_5_1_transcript.docx": "https://www.youtube.com/watch?v=WAiLM7OFU9A",
    "video_C6_5_2_transcript.docx": "https://www.youtube.com/watch?v=UfLLBOPvgOU",
    "video_C7_1_transcript.docx": "https://www.youtube.com/watch?v=_KvZN8-SyvQ&t=929s",
    "video_C7_2_1_transcript.docx": "https://www.youtube.com/watch?v=TqKBlC-zyKY&t=392s",
    "video_C7_2_2_transcript.docx": "https://www.youtube.com/watch?v=ptwSPTt2XnM&t=386s",
    "video_C7_3_1_transcript.docx": "https://www.youtube.com/watch?v=8-3xv_NElG0&t=137s",
    "video_C7_3_2_transcript.docx": "https://www.youtube.com/watch?v=IKD0O35NOUI",
    "video_C8_1_1_transcript.docx": "https://www.youtube.com/watch?v=qJj_LY1r91U&t=524s",
    "video_C8_1_2_transcript.docx": "https://www.youtube.com/watch?v=_Km_A2iRUds&t=631s",
    "video_C8_2_transcript.docx": "https://www.youtube.com/watch?v=_Cu7kGoRaE0",
    "video_C8_3_transcript.docx": "https://www.youtube.com/watch?v=KjPEqyGCtUs",
    "video_C8_4_1_transcript.docx": "https://www.youtube.com/watch?v=0DGe4fjr1aw&t=2s",
    "video_C8_4_2_transcript.docx": "https://www.youtube.com/watch?v=wKMBVF_bJdw",
    "video_C9_1_1_transcript.docx": "https://www.youtube.com/watch?v=4EdX3Ga9YoM&t=647s",
    "video_C9_1_2_transcript.docx": "https://www.youtube.com/watch?v=--JpgsDEL40",
    "video_C9_2_1_transcript.docx": "https://www.youtube.com/watch?v=ROIgZ5tyDFo&t=216s",
    "video_C9_2_2_transcript.docx": "https://www.youtube.com/watch?v=S8__bXkLSbM",
    "video_C9_3_transcript.docx": "https://www.youtube.com/watch?v=my3qRjVJ7VM",
    "video_C10_1_transcript.docx": "https://www.youtube.com/watch?v=1tCmeHf1Xk0&t=399s",
    "video_C10_2_transcript.docx": "https://www.youtube.com/watch?v=5DE5HXG8FWk&t=431s",
    "video_C10_3_transcript.docx": "https://www.youtube.com/watch?v=NsWX_5oV8bY&t=226s",
    "video_C10_4_1_transcript.docx": "https://www.youtube.com/watch?v=UXxELgk5Vws&t=575s",
    "video_C10_4_2_transcript.docx": "https://www.youtube.com/watch?v=JGxo_olUl2U&t=152s",
    "video_C10_4_3_transcript.docx": "https://www.youtube.com/watch?v=7AZr_li6ZtA&t=634s",
    "video_C10_5_transcript.docx": "https://www.youtube.com/watch?v=jKnjyvvXzXI&t=810s",
    "video_C10_6_transcript.docx": "https://www.youtube.com/watch?v=fEGw6eEre2I&t=477s",
    "video_C10_7_transcript.docx": "https://www.youtube.com/watch?v=iMfkIHkU6NM&t=1136s",
}

    # --- CHẠY QUY TRÌNH ---
    all_chunks = process_directory(VIDEO_MAP, TRANSCRIPT_DIR)

    # --- KIỂM TRA KẾT QUẢ ---
    if all_chunks:
        print("\n--- Dữ liệu mẫu (Chunk đầu tiên) ---")
        import json
        print(json.dumps(all_chunks[0], indent=2, ensure_ascii=False))

        print("\n--- Dữ liệu mẫu (Chunk cuối cùng) ---")
        print(json.dumps(all_chunks[-1], indent=2, ensure_ascii=False))
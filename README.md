
# RAG Q&A Project

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline for answering questions based on external documents, specifically YouTube videos. Users can input YouTube links, extract transcripts, store embeddings in a database, and perform question answering using a RAG model.

---

## Features

- Accepts a list of YouTube links as input
- Extracts video transcripts for retrieval
- Stores embeddings in a local or cloud-based vector database
- Performs question answering using a RAG pipeline
- Fully modular and easy to extend

---

## Project Structure

```
rag-qa-project/
│
├─ data/                   # Raw data and processed transcripts
│   ├─ transcripts/        # Extracted transcripts from YouTube
│   └─ embeddings/         # Embeddings saved for retrieval
│
├─ src/                    # Source code
│   ├─ input_handler.py    # Functions to process YouTube links
│   ├─ db_setup.py         # Initialize vector database
│   ├─ rag_model.py        # RAG model initialization and QA
│   └─ config.py           # Project parameters and settings
│
├─ notebooks/              # Optional: Jupyter notebooks for experimentation
│
├─ requirements.txt        # Python dependencies
├─ README.md               # Project documentation
└─ main.py                 # Main script to run the pipeline
```

---

## Step 1: Prepare Input

Provide a list of YouTube video links. The system will:

1. Download transcripts (via YouTube API or `youtube-transcript-api`)
2. Preprocess the text (cleaning, splitting into chunks)

Example:

```python
from src.input_handler import process_youtube_links

links = [
    "https://www.youtube.com/watch?v=abcd1234",
    "https://www.youtube.com/watch?v=wxyz5678"
]

documents = process_youtube_links(links)
```

---

## Step 2: Set Parameters

Configure project parameters in `config.py`:

```python
# config.py
CHUNK_SIZE = 500
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_MODEL = "facebook/rag-token-nq"
DB_TYPE = "FAISS"  # Options: FAISS, Milvus, etc.
```

---

## Step 3: Initialize Database

Create a vector database to store embeddings for retrieval:

```python
from src.db_setup import init_vector_db
from src.input_handler import embed_documents

db = init_vector_db(db_type="FAISS")
embeddings = embed_documents(documents)
db.add_documents(embeddings)
```

---

## Step 4: Initialize RAG and Answer Questions

Once the database is ready, initialize the RAG model and ask questions:

```python
from src.rag_model import RAGQA

rag = RAGQA(db=db, model_name="facebook/rag-token-nq")

question = "What is RAG in NLP?"
answer = rag.answer_question(question)

print("Q:", question)
print("A:", answer)
```

---

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Example packages:

- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `youtube-transcript-api`
- `torch`

---

## Notes

- The input handler can be extended to accept PDFs, articles, or other sources.
- The database can be replaced with cloud solutions for scalability.
- Chunk size and embedding model can be tuned for better retrieval performance.

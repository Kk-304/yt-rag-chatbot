# YouTube RAG Chatbot

A **Retrieval-Augmented Generation** chatbot that answers questions about any YouTube video by grounding responses in its transcript. Built with **LangChain**, **Google Gemini**, **FAISS**, and **FastAPI**, with a minimal HTML frontend.

## Live demo

https://yt-rag-chatbot-irvl.onrender.com/

The root URL serves a simple chat UI; `/docs` exposes interactive API documentation.

## How it works

1. **Transcript ingestion** — `youtube-transcript-api` fetches the video's caption text.
2. **Chunking** — `RecursiveCharacterTextSplitter` splits the transcript into 1,000-char chunks with 200-char overlap.
3. **Embedding** — Google `gemini-embedding-001` converts each chunk to a vector.
4. **Indexing** — vectors stored in an in-memory **FAISS** index, cached per-video.
5. **Retrieval** — for each question, the top 4 most relevant chunks are retrieved.
6. **Generation** — `gemini-2.5-flash` answers using only the retrieved context, with an explicit anti-hallucination instruction.

## Stack

| Layer        | Tool                                     |
|--------------|------------------------------------------|
| Frontend     | Vanilla HTML / CSS / JS (no framework)   |
| Backend      | FastAPI + Uvicorn                        |
| Orchestration| LangChain (LCEL)                         |
| Embeddings   | Google `text-embedding-004`              |
| LLM          | Google `gemini-2.5-flash`                |
| Vector store | FAISS (in-process, in-memory)            |
| Transcripts  | `youtube-transcript-api` v1              |
| Hosting      | Render (free web service)                |

## API

### `POST /ask`
Request:
```json
{ "video_id": "J5_-l7WIO_w", "question": "What is RAG?" }
```
Response:
```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "video_id": "J5_-l7WIO_w",
  "latency_seconds": 2.41,
  "cache_hit": false
}
```

### `DELETE /cache/{video_id}`
Drops a cached vector store so the next `/ask` rebuilds it.

### `GET /health`
Returns server status.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env       # then add your Google API key (from aistudio.google.com) to .env
uvicorn app:app --reload
```
Open http://localhost:8000

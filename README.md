# YouTube RAG Chatbot

A **Retrieval-Augmented Generation** chatbot that answers questions about any YouTube video by grounding responses in its transcript. Built with **LangChain**, **OpenAI**, **FAISS**, and **FastAPI**, with a minimal HTML frontend.

## Live demo

> Replace this with your deployed URL once Render finishes the first build:
> `https://yt-rag-chatbot.onrender.com`

The root URL serves a simple chat UI; `/docs` exposes interactive API documentation.

## How it works

1. **Transcript ingestion** — `youtube-transcript-api` fetches the video's caption text.
2. **Chunking** — `RecursiveCharacterTextSplitter` splits the transcript into 1,000-char chunks with 200-char overlap.
3. **Embedding** — OpenAI `text-embedding-3-small` converts each chunk to a vector.
4. **Indexing** — vectors stored in an in-memory **FAISS** index, cached per-video.
5. **Retrieval** — for each question, the top 4 most relevant chunks are retrieved.
6. **Generation** — `gpt-4o-mini` answers using only the retrieved context, with an explicit anti-hallucination instruction.

## Stack

| Layer        | Tool                                     |
|--------------|------------------------------------------|
| Frontend     | Vanilla HTML / CSS / JS (no framework)   |
| Backend      | FastAPI + Uvicorn                        |
| Orchestration| LangChain (LCEL)                         |
| Embeddings   | OpenAI `text-embedding-3-small`          |
| LLM          | OpenAI `gpt-4o-mini`                     |
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
cp .env.example .env       # then add your OpenAI API key to .env
uvicorn app:app --reload
```
Open http://localhost:8000/docs

## Evaluation

A small evaluation harness is included for measuring quality:
```bash
python eval.py --url http://localhost:8000 --testset test_set.example.json
```
This writes `eval_results.json`, which you mark as correct/incorrect to compute accuracy + average latency.

"""FastAPI wrapper for the YouTube RAG chatbot."""
import os
import time
import pathlib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import build_pipeline_for_video

load_dotenv()

app = FastAPI(
    title="YouTube RAG Chatbot",
    description="Ask questions about any YouTube video grounded in its transcript.",
    version="1.0.0",
)

# Allow any origin so the public Swagger UI / curl / a frontend can hit it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-video chain cache — avoids re-embedding the same transcript every question.
# In-memory (resets on server restart). Fine for a portfolio demo.
_chain_cache: dict = {}


class AskRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video ID, e.g. 'J5_-l7WIO_w'")
    question: str = Field(..., description="Question about the video")


class AskResponse(BaseModel):
    answer: str
    video_id: str
    latency_seconds: float
    cache_hit: bool


_STATIC_DIR = pathlib.Path(__file__).parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"
_HAS_FRONTEND = _INDEX_HTML.is_file()

if _HAS_FRONTEND:
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root():
    """Serve the HTML frontend if present, else a JSON pointer to the API docs."""
    if _HAS_FRONTEND:
        return FileResponse(_INDEX_HTML)
    return {
        "status": "ok",
        "message": "Frontend not found. Use /docs for the interactive API.",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY"))}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask a question about a YouTube video.

    First call for a video_id: fetches transcript, embeds, and caches the chain.
    Subsequent calls reuse the cached chain (much faster).
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(500, "Server is missing GOOGLE_API_KEY.")

    start = time.time()
    cache_hit = req.video_id in _chain_cache

    if not cache_hit:
        try:
            _chain_cache[req.video_id] = build_pipeline_for_video(req.video_id)
        except Exception as e:
            raise HTTPException(400, f"Could not build pipeline: {e}")

    chain = _chain_cache[req.video_id]
    try:
        answer = chain.invoke(req.question)
    except Exception as e:
        raise HTTPException(500, f"LLM call failed: {e}")

    return AskResponse(
        answer=answer,
        video_id=req.video_id,
        latency_seconds=round(time.time() - start, 2),
        cache_hit=cache_hit,
    )


@app.delete("/cache/{video_id}")
def clear_cache(video_id: str):
    """Drop a cached chain so the next /ask rebuilds it."""
    existed = _chain_cache.pop(video_id, None) is not None
    return {"video_id": video_id, "cleared": existed}

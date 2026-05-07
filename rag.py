"""Core RAG pipeline: fetch a YouTube transcript and build a retrieval chain over it.

Uses Google Gemini for both embeddings and generation (free tier, no credit card).
"""
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
EMBEDDING_MODEL = "gemini-embedding-001"   # Google embedding model, free tier
LLM_MODEL = "gemini-2.5-flash"                  # Free tier: ~10 RPM, ~250 RPD
CHUNK_SIZE = 1000                               # characters per chunk
CHUNK_OVERLAP = 200                             # so context isn't cut mid-thought
TOP_K = 4                                       # chunks retrieved per question


def fetch_transcript(video_id: str, languages=("en",)) -> str:
    """Fetch transcript text for a YouTube video. Returns a single joined string."""
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=list(languages))
    except Exception:
        # Fallback: list all available transcripts and grab the first one
        transcript_list = api.list(video_id)
        available = [t.language_code for t in transcript_list]
        if not available:
            raise
        fetched = api.fetch(video_id, languages=available)
    return " ".join(snippet.text for snippet in fetched.snippets)


def build_vector_store(transcript: str) -> FAISS:
    """Chunk the transcript and index Gemini embeddings into a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = splitter.create_documents([transcript])
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(docs, embeddings)


def build_chain(vector_store: FAISS, k: int = TOP_K):
    """Build the LCEL retrieval chain: question → retrieved context → grounded answer."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant answering questions about a YouTube video.\n"
            "Use ONLY the transcript context below. If the answer isn't in the context, "
            "say you don't know \u2014 do not make things up.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

    return (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )


def build_pipeline_for_video(video_id: str):
    """Convenience: transcript \u2192 vector store \u2192 chain. Returns the chain."""
    transcript = fetch_transcript(video_id)
    vs = build_vector_store(transcript)
    return build_chain(vs)

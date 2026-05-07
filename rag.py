"""Core RAG pipeline: fetch a YouTube transcript and build a retrieval chain over it."""
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
EMBEDDING_MODEL = "text-embedding-3-small"  # ~$0.02 per 1M tokens
LLM_MODEL = "gpt-4o-mini"                   # cheap, fast, accurate enough for RAG
CHUNK_SIZE = 1000                           # characters per chunk
CHUNK_OVERLAP = 200                         # overlap so context isn't cut mid-thought
TOP_K = 4                                   # how many chunks to retrieve per question


def fetch_transcript(video_id: str, languages=("en",)) -> str:
    """Fetch transcript text for a YouTube video. Returns a single joined string."""
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=list(languages))
    # FetchedTranscript object → join all snippet texts
    return " ".join(snippet.text for snippet in fetched.snippets)


def build_vector_store(transcript: str) -> FAISS:
    """Chunk the transcript and index embeddings into a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = splitter.create_documents([transcript])
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(docs, embeddings)


def build_chain(vector_store: FAISS, k: int = TOP_K):
    """Build the LCEL retrieval chain: question → retrieved context → grounded answer."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant answering questions about a YouTube video.\n"
            "Use ONLY the transcript context below. If the answer isn't in the context, "
            "say you don't know — do not make things up.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

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
    """Convenience: transcript → vector store → chain. Returns the chain."""
    transcript = fetch_transcript(video_id)
    vs = build_vector_store(transcript)
    return build_chain(vs)

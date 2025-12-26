"""FastAPI API for Confidential Interrogation Records RAG chatbot."""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Optional, Any
import uuid

from backend.RAG_pipeline import initialize_rag
from backend.logger import get_logger
from backend import config
from backend.models.models import ChatRequest, ChatResponse

log = get_logger(__name__)

graph: Optional[Any] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for FastAPI.

    Handles initialization and cleanup:
    - On startup: Initialize the RAG system and store the workflow graph
    - On shutdown: Clean up resources
    """
    global graph

    log.info("Starting API server...")
    graph = initialize_rag()  # Initialize complete RAG pipeline
    log.info("API ready!")

    yield  # Application runs here

    log.info("Shutting down API server...")  # Cleanup would go here if needed


app = FastAPI(
    title="Confidential Interrogation Records RAG API",
    description="Offline RAG system for confidential police interrogation records",
    version="1.0.0",
    lifespan=lifespan  # Use lifespan manager for initialization
)


app = FastAPI(
    title="Confidential Interrogation Records RAG API",
    description="Offline RAG system for confidential police interrogation records",
    version="1.0.0",
    lifespan=lifespan
)


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Confidential Interrogation Records RAG API is running",
        "model": config.OLLAMA_MODEL,
        "embedding_model": config.OLLAMA_EMBEDDING_MODEL
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message through the RAG pipeline and return AI response.

    This endpoint:
    1. Accepts user messages with optional conversation thread IDs
    2. Processes queries through the complete RAG pipeline (reformulate → retrieve → generate)
    3. Maintains conversation context using LangGraph checkpointer
    4. Returns AI-generated responses with thread continuity

    Args:
        request: ChatRequest containing message and optional thread_id

    Returns:
        ChatResponse: AI response with thread_id for conversation continuity

    Raises:
        HTTPException: If RAG system is not initialized (503) or processing fails (500)
    """
    if graph is None:
        raise HTTPException(status_code=503, detail="RAG graph not initialized")

    # Use provided thread_id or generate new one for conversation tracking
    thread_id = request.thread_id or str(uuid.uuid4())
    graph_config: Any = {"configurable": {"thread_id": thread_id}}

    try:
        # Invoke the LangGraph workflow with user query
        # The graph handles: query reformulation → document retrieval → response generation
        result = graph.invoke(
            {
                "user_query": request.message,  # User's input message
                "context": ""  # Will be populated by retrieval step
            },
            config=graph_config  # Thread-specific configuration for conversation persistence
        )

        # Extract the final AI response from the workflow result
        response_text = result["messages"][-1].content
        log.info(f"Chat processed for thread {thread_id}: query='{request.message[:50]}...' response_length={len(response_text)}")
        return ChatResponse(response=response_text, thread_id=thread_id)

    except Exception as e:
        log.error(f"Error processing chat for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )

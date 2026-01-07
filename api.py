"""
FastAPI server for the data analysis agent.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
from dotenv import load_dotenv
from agent.root import DataAgent

load_dotenv()

app = FastAPI(
    title="SynMax Data Agent API",
    description="API for natural language data analysis",
    version="1.0.0"
)

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
dataset_path = os.getenv("DATASET_PATH", "./data/dataset.csv")
agent = DataAgent(dataset_path=dataset_path, model="gpt-5-nano")


class QueryRequest(BaseModel):
    """Request model for queries."""
    question: str
    dataset_path: Optional[str] = None
    thread_id: Optional[str] = "default"


class Message(BaseModel):
    """Message model for conversation history."""
    role: str
    content: str


class QueryResponse(BaseModel):
    """Response model for queries."""
    question: str
    answer: str
    conversation: List[Message]
    metadata: Dict[str, Any]
    status: str = "success"


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SynMax Data Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Submit a data analysis question (batch mode)",
            "POST /query/stream": "Stream a data analysis question (real-time)",
            "GET /health": "Health check",
            "GET /dataset/info": "Get dataset information",
            "GET /docs": "API documentation"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "dataset_configured": os.path.exists(agent.dataset_path),
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY"))
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a natural language query about the dataset (batch mode).

    Args:
        request: QueryRequest containing the question and optional thread_id

    Returns:
        QueryResponse with answer, conversation history, and metadata
    """
    try:
        # Use custom dataset path if provided
        if request.dataset_path:
            query_agent = DataAgent(dataset_path=request.dataset_path)
        else:
            query_agent = agent

        # Process the query
        result = query_agent.query(
            question=request.question,
            thread_id=request.thread_id
        )

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            conversation=[Message(**msg) for msg in result["conversation"]],
            metadata=result["metadata"],
            status="success"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Stream a natural language query about the dataset (real-time mode).

    Args:
        request: QueryRequest containing the question and optional thread_id

    Returns:
        StreamingResponse with Server-Sent Events (SSE) format
    """
    try:
        # Use custom dataset path if provided
        if request.dataset_path:
            query_agent = DataAgent(dataset_path=request.dataset_path)
        else:
            query_agent = agent

        async def event_generator():
            """Generate SSE events from agent stream."""
            for event in query_agent.stream(
                question=request.question,
                thread_id=request.thread_id
            ):
                # Format as SSE
                yield f"data: {json.dumps(event)}\n\n"

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'content': 'Stream finished'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error streaming query: {str(e)}"
        )


@app.get("/dataset/info")
async def dataset_info():
    """Get information about the loaded dataset."""
    if not os.path.exists(agent.dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found at {agent.dataset_path}"
        )

    # TODO: Implement actual dataset info extraction
    return {
        "path": agent.dataset_path,
        "exists": True,
        "info": "Dataset info will be implemented"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

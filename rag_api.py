from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from rag_with_llm import RAGWithLLM
import time
from pathlib import Path
import os

app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation API")

DOCUMENTS_DIR = Path("documents")
DOCUMENTS_DIR.mkdir(exist_ok=True)

rag = None

def get_rag():
    global rag
    if rag is None:
        llm_type = os.getenv("LLM_TYPE", "mock")
        lazy_init = os.getenv("RAG_LAZY_INIT", "1") == "1"
        print("Loading RAG system...")
        rag = RAGWithLLM(llm_type=llm_type, load_vectorstore_on_init=not lazy_init)
        print("✓ RAG system ready")
    return rag

# Request/Response models
class Question(BaseModel):
    query: str
    k: Optional[int] = 3

class Citation(BaseModel):
    source: str
    page: str
    relevance: float
    excerpt: str

class Answer(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    response_time_ms: float

class IngestResponse(BaseModel):
    file_name: str
    chunks_added: int

@app.get("/")
async def root():
    return {
        "service": "RAG API",
        "endpoints": {
            "POST /ask": "Ask a question",
            "GET /health": "Health check",
            "GET /sources": "List available sources"
        }
    }

@app.post("/ask", response_model=Answer)
async def ask(question: Question):
    """Ask a question and get answer with citations."""
    
    start_time = time.time()
    
    try:
        rag_instance = get_rag()
        if not rag_instance.rag.vectorstore:
            raise HTTPException(status_code=400, detail="Vector store not initialized. Run rag_core.py or ingest documents first.")
        result = rag_instance.ask(question.query, k=question.k)
        
        response_time = (time.time() - start_time) * 1000
        
        citations = []
        for citation in result["citations"]:
            citations.append({
                "source": citation.get("source", "unknown"),
                "page": str(citation.get("page", "N/A")),
                "relevance": float(citation.get("relevance", 0.0)),
                "excerpt": citation.get("excerpt", "")
            })

        return Answer(
            question=result["question"],
            answer=result["answer"],
            citations=citations,
            response_time_ms=round(response_time, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"/ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    llm_type = os.getenv("LLM_TYPE", "mock")
    return {"status": "healthy", "llm_type": llm_type}

@app.get("/sources")
async def list_sources():
    """List available document sources."""
    sources = []
    for file_path in DOCUMENTS_DIR.glob("*"):
        if file_path.suffix.lower() in {".txt", ".pdf"}:
            sources.append(str(file_path))
    return {"sources": sorted(sources)}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a document into the vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")

    save_path = DOCUMENTS_DIR / Path(file.filename).name
    content = await file.read()
    save_path.write_bytes(content)

    chunks_added = rag.rag.add_documents([str(save_path)])
    return IngestResponse(file_name=save_path.name, chunks_added=chunks_added)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
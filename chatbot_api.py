from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from contextlib import asynccontextmanager
from functools import partial

import sqlite3
import time as _time

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DB_DIR = "./chroma_db"
DOCSTORE_PATH = "./docstore.json"
ANALYTICS_DB  = "./analytics.db"


# ─── Analytics SQLite ────────────────────────────────────
def init_analytics_db():
    """Create analytics tables if they don't exist."""
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            mode        TEXT    NOT NULL,       -- 'agent' or 'qa'
            question    TEXT    NOT NULL,
            response_ms INTEGER NOT NULL DEFAULT 0,
            tool_count  INTEGER NOT NULL DEFAULT 0,
            success     INTEGER NOT NULL DEFAULT 1
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS tool_usage (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            tool_name   TEXT    NOT NULL,
            request_id  INTEGER,
            FOREIGN KEY (request_id) REFERENCES requests(id)
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Analytics DB initialized.")


def log_request(mode: str, question: str, response_ms: int, tool_count: int = 0, success: bool = True) -> int:
    """Log a request and return its ID."""
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO requests (timestamp, mode, question, response_ms, tool_count, success) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), mode, question, response_ms, tool_count, 1 if success else 0)
    )
    req_id = c.lastrowid
    conn.commit()
    conn.close()
    return req_id


def log_tool_usage(tool_names: list, request_id: int):
    """Log which tools were called for a request."""
    if not tool_names:
        return
    conn = sqlite3.connect(ANALYTICS_DB)
    c = conn.cursor()
    ts = datetime.now().isoformat()
    for name in tool_names:
        c.execute(
            "INSERT INTO tool_usage (timestamp, tool_name, request_id) VALUES (?, ?, ?)",
            (ts, name, request_id)
        )
    conn.commit()
    conn.close()

# Global variables
qa_system    = None
vector_db    = None
llm_model    = None
parent_store = {}   # { parent_id: parent_text }
bm25_index   = None # BM25 keyword index over child chunks
bm25_docs    = []   # child docs in same order as bm25_index
react_agent  = None # ReAct agent for agentic workflows


# --- Initialize System ---
def initialize_qa_system():
    """Load LLM and existing ChromaDB built by ingest.py"""
    global qa_system, vector_db, llm_model, parent_store, bm25_index, bm25_docs, react_agent

    # Deferred imports to avoid loading PyTorch/transformers at server startup
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    try:
        if not os.getenv("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY environment variable not set")
            return False

        llm_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.5,
            max_retries=0,
        )

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        # Load pre-built ChromaDB (created by ingest.py)
        if not os.path.exists(CHROMA_DB_DIR):
            logger.error(
                f"No ChromaDB found at '{CHROMA_DB_DIR}'. "
                "Run ingest.py first to build the database."
            )
            return False

        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding_model,
        )

        count = vector_db._collection.count()
        if count == 0:
            logger.error("ChromaDB exists but is empty. Re-run ingest.py.")
            vector_db = None
            return False

        logger.info(f"Loaded ChromaDB with {count} child chunks.")

        # Load parent docstore
        if os.path.exists(DOCSTORE_PATH):
            with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
                parent_store = json.load(f)
            logger.info(f"Loaded parent docstore with {len(parent_store)} parent chunks.")
        else:
            logger.warning(
                f"No docstore found at '{DOCSTORE_PATH}'. "
                "Re-run ingest.py to enable Parent-Child Chunking."
            )

        # Build BM25 index from child chunks
        try:
            from rank_bm25 import BM25Okapi
            all_child_data = vector_db._collection.get(include=["documents", "metadatas"])
            bm25_docs = [
                type("Doc", (), {"page_content": txt, "metadata": meta})()
                for txt, meta in zip(all_child_data["documents"], all_child_data["metadatas"])
            ]
            tokenized = [doc.page_content.lower().split() for doc in bm25_docs]
            bm25_index = BM25Okapi(tokenized)
            logger.info(f"Built BM25 index over {len(bm25_docs)} child chunks.")
        except Exception as e:
            logger.warning(f"BM25 index build failed (will use vector-only): {e}")

        # Build ReAct agent
        try:
            from agent import build_agent
            react_agent = build_agent(llm_model)
            logger.info("ReAct agent built successfully.")
        except Exception as e:
            logger.warning(f"ReAct agent build failed: {e}")

        qa_system = create_qa_chain(vector_db, llm_model)
        return True

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return False


def create_qa_chain(db, llm):
    """Create QA graph with LangGraph for chat history support"""
    from langgraph.graph import StateGraph, END
    from langchain_core.prompts import PromptTemplate
    from sentence_transformers import CrossEncoder
    from typing import TypedDict, List, Any

    if not db or not llm:
        return None

    # --- Shared State ---
    class GraphState(TypedDict):
        question: str
        history: str
        active_entity: str
        docs: List[Any]
        reranked_docs: List[Any]
        context: str
        answer: str

    vector_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.4}
    )

    def hybrid_retrieve(query: str, k: int = 8):
        """Combine BM25 + vector search using Reciprocal Rank Fusion."""
        from langchain_core.documents import Document

        # --- Vector results ---
        vector_results = vector_retriever.invoke(query)

        # --- BM25 results ---
        bm25_results = []
        if bm25_index and bm25_docs:
            tokens = query.lower().split()
            scores = bm25_index.get_scores(tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            bm25_results = [bm25_docs[i] for i in top_indices if scores[i] > 0]

        # --- RRF merge ---
        rrf_scores: dict[str, float] = {}
        doc_map:    dict[str, object] = {}

        def rrf(rank: int, k: int = 60) -> float:
            return 1.0 / (rank + k)

        for rank, doc in enumerate(vector_results):
            key = doc.page_content[:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf(rank)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_results):
            key = doc.page_content[:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf(rank)
            doc_map[key] = doc

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [doc_map[key] for key, _ in ranked]

    try:
        reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')
    except Exception as e:
        logger.warning(f"Failed to load reranker: {e}")
        reranker = None

    qa_prompt = PromptTemplate.from_template(
        """You are a helpful ERP assistant. Answer the question based on the context below.

Language Rule:
- Detect the language of the Question only.
- If the question is in Arabic, reply entirely in Arabic.
- If the question is in English, reply entirely in English.
- Never mix languages.

Answer Rules:
- Answer ONLY from the Context provided. Do NOT use outside knowledge.
- Active entity is: {active_entity}
- If the context does not contain enough information, say: "المعلومات غير متوفرة في الوثائق الحالية" (Arabic) or "This information is not available in the current documents." (English).
- Never invent steps, procedures, or details not found in the context.

Conversation History:
{history}

Context:
{context}

Question: {question}

Answer:"""
    )

    # --- Node 1: Retrieve ---
    def retrieve_node(state: GraphState) -> GraphState:
        logger.info("Graph node: retrieve")
        from langchain_core.documents import Document

        child_docs = hybrid_retrieve(state["question"])

        # Swap each child doc for its parent (larger context)
        if parent_store:
            seen = set()
            parent_docs = []
            for doc in child_docs:
                pid = doc.metadata.get("parent_id")
                if pid and pid not in seen and pid in parent_store:
                    parent_docs.append(Document(page_content=parent_store[pid]))
                    seen.add(pid)
                elif not pid:
                    parent_docs.append(doc)   # fallback: keep child as-is
            docs = parent_docs if parent_docs else child_docs
            logger.info(f"  {len(child_docs)} child hits → {len(docs)} parent chunks")
        else:
            docs = child_docs  # no docstore — use child docs directly

        return {**state, "docs": docs}

    # --- Node 2: Rerank ---
    def rerank_node(state: GraphState) -> GraphState:
        logger.info("Graph node: rerank")
        docs = state["docs"]
        if not reranker:
            return {**state, "reranked_docs": docs[:4]}
        try:
            pairs = [(state["question"], doc.page_content) for doc in docs]
            scores = reranker.predict(pairs)
            reranked = [docs[i] for i in sorted(range(len(scores)),
                        key=lambda k: scores[k], reverse=True)][:4]
            return {**state, "reranked_docs": reranked}
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return {**state, "reranked_docs": docs[:4]}

    # --- Node 3: Generate ---
    def generate_node(state: GraphState) -> GraphState:
        logger.info("Graph node: generate")
        reranked_docs = state.get("reranked_docs", [])
        context = (
            "\n".join([d.page_content for d in reranked_docs])
            if reranked_docs else "No relevant context found."
        )
        prompt_text = qa_prompt.format(
            context=context,
            question=state["question"],
            history=state["history"],
            active_entity=state["active_entity"]
        )
        response = llm.invoke(prompt_text)
        answer = response.content if hasattr(response, "content") else str(response)
        return {**state, "context": context, "answer": answer}

    # --- Build Graph ---
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Document Q&A System...")
    init_analytics_db()
    success = initialize_qa_system()
    if success:
        logger.info("System ready.")
    else:
        logger.warning(
            "System started without a knowledge base. "
            "Run ingest.py to build the database, then restart."
        )
    yield
    # Shutdown
    logger.info("Shutting down Document Q&A System...")


# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A System",
    description="Ask questions about pre-loaded company documents using AI",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your website domain before going live
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str
    history: Optional[str] = ""
    active_entity: Optional[str] = "Purchases"


class QuestionResponse(BaseModel):
    answer: str
    context_used: str
    timestamp: datetime


class SystemStatus(BaseModel):
    database_initialized: bool
    document_count: int
    model_loaded: bool
    status: str


# --- API Endpoints ---

@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Document Q&A System API",
        "version": "2.0.0",
        "endpoints": {
            "ask": "/ask-question/",
            "status": "/status/",
            "health": "/health/",
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}


@app.get("/status", response_model=SystemStatus)
async def get_status():
    global qa_system, vector_db, llm_model

    database_initialized = vector_db is not None
    document_count = 0

    if vector_db:
        try:
            document_count = vector_db._collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")

    model_loaded = llm_model is not None

    if database_initialized and model_loaded:
        status = "ready"
    elif model_loaded:
        status = "model_loaded_no_documents"
    else:
        status = "not_initialized"

    return SystemStatus(
        database_initialized=database_initialized,
        document_count=document_count,
        model_loaded=model_loaded,
        status=status
    )


@app.post("/ask-question/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the pre-loaded documents"""
    global qa_system, vector_db

    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

    if not qa_system or not vector_db:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not loaded. Run ingest.py then restart the server."
        )

    try:
        chain_input = {
            "question": request.question,
            "history": request.history or "",
            "active_entity": request.active_entity or "Purchases"
        }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, partial(qa_system.invoke, chain_input))
        answer = result["answer"]
        context_used = result.get("context", "")[:400]

        return QuestionResponse(
            answer=answer,
            context_used=context_used,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Question processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")


@app.post("/ask-question/stream")
async def ask_question_stream(request: QuestionRequest):
    """Streaming version — tokens arrive as they're generated"""
    global qa_system, vector_db

    if not qa_system or not vector_db:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded.")

    async def generate() -> AsyncGenerator[str, None]:
        try:
            loop = asyncio.get_event_loop()
            chain_input = {
                "question": request.question,
                "history": request.history or "",
                "active_entity": request.active_entity or "Purchases",
            }

            t0 = _time.time()
            result = await loop.run_in_executor(None, partial(qa_system.invoke, chain_input))
            elapsed = int((_time.time() - t0) * 1000)
            log_request("qa", request.question, elapsed, 0, success=True)

            # Stream the answer in small chunks
            answer = result["answer"]
            for i in range(0, len(answer), 4):
                yield f"data: {json.dumps({'token': answer[i:i+4]})}\n\n"
                await asyncio.sleep(0)

            yield f"data: {json.dumps({'done': True, 'context': result.get('context', '')[:400]})}\n\n"

        except Exception as e:
            log_request("qa", request.question, 0, 0, success=False)
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


class AgentMessage(BaseModel):
    role: str    # "user" or "assistant"
    content: str


class AgentRequest(BaseModel):
    message: str
    history: Optional[List[AgentMessage]] = []


class AgentResponse(BaseModel):
    answer: str
    tool_calls: int
    timestamp: datetime


@app.post("/agent/", response_model=AgentResponse)
async def agent_endpoint(request: AgentRequest):
    """
    ReAct Agent endpoint — handles multi-step procurement workflows.
    Supports: purchase orders, suppliers, invoices.
    """
    global react_agent

    if not react_agent:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Check server logs."
        )

    try:
        from agent import run_agent
        history = [{"role": m.role, "content": m.content} for m in (request.history or [])]
        loop   = asyncio.get_event_loop()
        t0 = _time.time()
        result = await loop.run_in_executor(None, partial(run_agent, react_agent, request.message, history))
        elapsed = int((_time.time() - t0) * 1000)

        # Log analytics
        req_id = log_request("agent", request.message, elapsed, result["tool_calls"], success=True)
        if result.get("tools_used"):
            log_tool_usage(result["tools_used"], req_id)

        return AgentResponse(
            answer=result["answer"],
            tool_calls=result["tool_calls"],
            timestamp=datetime.now()
        )
    except Exception as e:
        log_request("agent", request.message, 0, 0, success=False)
        logger.error(f"Agent endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Nightly Stock Check Endpoint ────────────────────────

@app.get("/stock-check")
async def stock_check():
    """
    Nightly stock check — returns all items below minimum level
    with recommended order quantities grouped by supplier.
    Call this from a cron job (e.g. 2am nightly) or manually.
    In production, this triggers WhatsApp/email alerts.
    """
    try:
        from agent import MOCK_INVENTORY, MOCK_RECIPES, MOCK_DAILY_SALES
        alerts = []
        total_cost = 0.0
        supplier_groups = {}

        for item in MOCK_INVENTORY:
            if item["current_stock"] < item["min_level"]:
                deficit = item["min_level"] - item["current_stock"]
                order_qty = round(deficit * 1.5)
                cost = round(order_qty * item["cost_per_unit"], 2)
                total_cost += cost
                supplier = item["supplier"]
                if supplier not in supplier_groups:
                    supplier_groups[supplier] = []
                supplier_groups[supplier].append({
                    "item": item["name"],
                    "current": item["current_stock"],
                    "minimum": item["min_level"],
                    "order_qty": order_qty,
                    "unit": item["unit"],
                    "cost": cost
                })
                alerts.append(item["name"])

        return {
            "timestamp": datetime.now().isoformat(),
            "low_stock_count": len(alerts),
            "total_estimated_cost": round(total_cost, 2),
            "low_stock_items": alerts,
            "orders_by_supplier": supplier_groups,
            "whatsapp_status": "mock — would send alert to manager",
            "message": f"تنبيه: {len(alerts)} أصناف تحت الحد الأدنى" if alerts else "✅ جميع الأصناف فوق الحد الأدنى"
        }
    except Exception as e:
        logger.error(f"Stock check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Analytics Endpoints ─────────────────────────────────

@app.get("/analytics/summary")
async def analytics_summary():
    """Dashboard summary: total requests, avg response time, success rate, mode split."""
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as total FROM requests")
    total = c.fetchone()["total"]

    c.execute("SELECT AVG(response_ms) as avg_ms FROM requests WHERE success=1")
    avg_ms = c.fetchone()["avg_ms"] or 0

    c.execute("SELECT COUNT(*) as cnt FROM requests WHERE success=1")
    success_cnt = c.fetchone()["cnt"]

    c.execute("SELECT COUNT(*) as cnt FROM requests WHERE mode='agent'")
    agent_cnt = c.fetchone()["cnt"]

    c.execute("SELECT COUNT(*) as cnt FROM requests WHERE mode='qa'")
    qa_cnt = c.fetchone()["cnt"]

    # Today's count
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) as cnt FROM requests WHERE timestamp LIKE ?", (today + "%",))
    today_cnt = c.fetchone()["cnt"]

    conn.close()
    return {
        "total_requests": total,
        "today_requests": today_cnt,
        "avg_response_ms": round(avg_ms),
        "success_rate": round((success_cnt / total * 100) if total > 0 else 0, 1),
        "agent_requests": agent_cnt,
        "qa_requests": qa_cnt,
    }


@app.get("/analytics/tools")
async def analytics_tools():
    """Tool usage breakdown."""
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT tool_name, COUNT(*) as count
        FROM tool_usage
        GROUP BY tool_name
        ORDER BY count DESC
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return {"tools": rows}


@app.get("/analytics/timeline")
async def analytics_timeline():
    """Requests per day for the last 30 days."""
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT DATE(timestamp) as day, COUNT(*) as count,
               SUM(CASE WHEN mode='agent' THEN 1 ELSE 0 END) as agent,
               SUM(CASE WHEN mode='qa' THEN 1 ELSE 0 END) as qa
        FROM requests
        WHERE timestamp >= DATE('now', '-30 days')
        GROUP BY DATE(timestamp)
        ORDER BY day
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return {"timeline": rows}


@app.get("/analytics/top-questions")
async def analytics_top_questions():
    """Most asked questions (top 15)."""
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT question, mode, COUNT(*) as count, AVG(response_ms) as avg_ms
        FROM requests
        GROUP BY question
        ORDER BY count DESC
        LIMIT 15
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return {"questions": rows}


@app.get("/analytics/recent")
async def analytics_recent():
    """Last 20 requests."""
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT id, timestamp, mode, question, response_ms, tool_count, success
        FROM requests
        ORDER BY id DESC
        LIMIT 20
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return {"recent": rows}


if __name__ == "__main__":
    import uvicorn

    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set.")
        print("Set it with: set GOOGLE_API_KEY=your_key_here")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )

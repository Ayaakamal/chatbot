from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import shutil
from datetime import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DB_DIR = "./chroma_db"

# Global variables
qa_system = None
vector_db = None
llm_model = None


# --- Initialize System ---
def initialize_qa_system():
    """Load LLM and existing ChromaDB built by ingest.py"""
    global qa_system, vector_db, llm_model

    # Deferred imports to avoid loading PyTorch/transformers at server startup
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    try:
        if not os.getenv("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY environment variable not set")
            return False

        # Initialize Gemini
        llm_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.5
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

        logger.info(f"Loaded ChromaDB with {count} chunks.")
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

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 40,
            "lambda_mult": 0.4
        }
    )

    try:
        reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')
    except Exception as e:
        logger.warning(f"Failed to load reranker: {e}")
        reranker = None

    qa_prompt = PromptTemplate.from_template(
        """You are a helpful AI assistant. Answer the question based on:
        1. The context below
        2. Our conversation history
        3. The current active entity: {active_entity}

        Language Rule:
        - Detect the language of the QUESTION only (ignore the language of the context).
        - Reply strictly in the same language as the QUESTION.
        - If the question is in English, reply in English. If Arabic, reply in Arabic. Never switch languages.

        Critical Instructions:
        - ALWAYS resolve "they", "their", "it" to the active entity
        - NEVER say "I don't know" for questions about the active entity
        - If context doesn't contain exact answer, infer from related information
        - Treat different terms for same concept as equivalent (e.g., 'location' and 'address')
        - Be aware that all questions are about {active_entity}. If the user asks any question, consider they are asking about {active_entity}, regardless of phrasing.

        Conversation History:
        {history}

        Context: {context}

        Question: {question}

        Answer:
        """
    )

    # --- Node 1: Retrieve ---
    def retrieve_node(state: GraphState) -> GraphState:
        logger.info("Graph node: retrieve")
        docs = retriever.invoke(state["question"])
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
    active_entity: Optional[str] = "Canva"


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
            "active_entity": request.active_entity or "EDGE-Pro"
        }

        result = qa_system.invoke(chain_input)
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

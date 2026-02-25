from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import tempfile
import shutil
from datetime import datetime
from contextlib import asynccontextmanager

# Heavy AI imports (LangChain, HuggingFace, Chroma, etc.) are deferred to avoid
# loading PyTorch at startup; they are imported inside initialize_qa_system and create_qa_chain.

# File processing imports
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import io

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
qa_system = None
vector_db = None
llm_model = None

# --- File Processing Functions ---
def process_uploaded_file(file_content: bytes, filename: str) -> str:
    """Process uploaded file and extract text"""
    text = ""
    try:
        if filename.endswith('.pdf'):
            pdf_reader = PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif filename.endswith(('.txt', '.md')):
            text += file_content.decode("utf-8") + "\n"
        elif filename.endswith('.docx'):
            doc = Document(io.BytesIO(file_content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(file_content))
            text += f"Parquet File: {filename}\n"
            text += df.to_string(index=False) + "\n\n"
            text += f"Data Summary:\n{df.describe().to_string()}\n"
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content))
            text += f"Excel File: {filename}\n"
            text += df.to_string(index=False) + "\n\n"
            text += f"Data Summary:\n{df.describe().to_string()}\n"
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
            text += f"CSV File: {filename}\n"
            text += df.to_string(index=False) + "\n\n"
            text += f"Data Summary:\n{df.describe().to_string()}\n"
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing {filename}: {str(e)}")
    
    return text

# --- Initialize System ---
def initialize_qa_system(documents_text: str = None):
    """Initialize the QA system with optional documents"""
    global qa_system, vector_db, llm_model

    # Deferred imports to avoid loading PyTorch/transformers at server startup
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    try:
        # Check if GOOGLE_API_KEY is set
        if not os.getenv("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY environment variable not set")
            return False
        
        # Initialize Gemini Pro
        llm_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.5
        )
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        # Process documents if provided
        if documents_text and documents_text.strip():
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                add_start_index=True
            )
            documents = text_splitter.create_documents([documents_text])
            
            # Create new Chroma instance
            vector_db = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory="./chroma_db"
            )
            logger.info("Documents processed and database created successfully!")
        else:
            # Check for existing Chroma DB
            if os.path.exists("./chroma_db"):
                try:
                    vector_db = Chroma(
                        persist_directory="./chroma_db",
                        embedding_function=embedding_model
                    )
                    # Verify database has content
                    if vector_db._collection.count() == 0:
                        logger.warning("Database exists but is empty")
                        vector_db = None
                except Exception as e:
                    logger.error(f"Error loading existing database: {e}")
                    vector_db = None
            else:
                vector_db = None
        
        # Create QA chain if database exists
        if vector_db:
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
        - Be aware that all questions are about EDGE-Pro company. If the user asks any question, consider they are asking about EDGE-Pro, regardless of phrasing.

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

# Lifespan event handler (modern FastAPI approach)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Document Q&A System...")
    try:
        success = initialize_qa_system()
        if success:
            logger.info("System initialized successfully")
        else:
            logger.warning("System initialization completed with warnings")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Q&A System...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Document Q&A System",
    description="Upload documents and ask questions about them using AI",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    history: Optional[str] = ""
    active_entity: Optional[str] = "EDGE-Pro"

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
    """Root endpoint with API information"""
    return {
        "message": "Document Q&A System API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload-documents/",
            "ask": "/ask-question/",
            "status": "/status/",
            "health": "/health/"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    global qa_system, vector_db, llm_model
    
    database_initialized = vector_db is not None
    document_count = 0
    
    if vector_db:
        try:
            document_count = vector_db._collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            document_count = 0
    
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

@app.post("/upload-documents/")
async def upload_documents(
    file1: Optional[UploadFile] = File(None, description="Document 1 (PDF, DOCX, TXT, etc.)"),
    file2: Optional[UploadFile] = File(None, description="Document 2"),
    file3: Optional[UploadFile] = File(None, description="Document 3"),
    file4: Optional[UploadFile] = File(None, description="Document 4"),
    file5: Optional[UploadFile] = File(None, description="Document 5"),
):
    """Upload and process documents. Use one or more of the file fields below."""
    files = [f for f in (file1, file2, file3, file4, file5) if f is not None and getattr(f, "filename", None)]
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded. Choose at least one file.")
    
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")
    
    try:
        all_text = ""
        processed_files = []
        
        for file in files:
            if file.size == 0:
                continue
                
            # Read file content
            content = await file.read()
            
            # Process file
            text = process_uploaded_file(content, file.filename)
            all_text += text + "\n\n"
            processed_files.append(file.filename)
        
        if not all_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from uploaded files")
        
        # Initialize system with documents
        success = initialize_qa_system(all_text)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize system with documents")
        
        return {
            "message": "Documents uploaded and processed successfully",
            "processed_files": processed_files,
            "total_files": len(processed_files),
            "text_length": len(all_text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ask-question/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded documents"""
    global qa_system, vector_db
    
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")
    
    if not qa_system or not vector_db:
        raise HTTPException(
            status_code=400, 
            detail="No documents uploaded. Please upload documents first using /upload-documents/"
        )
    
    try:
        # Prepare input for QA chain
        chain_input = {
            "question": request.question,
            "history": request.history or "",
            "active_entity": request.active_entity or "EDGE-Pro"
        }
        
        # Get answer from QA graph
        result = qa_system.invoke(chain_input)
        answer = result["answer"]

        # Get context used (for transparency)
        context_used = result.get("context", "")[:400]
        
        return QuestionResponse(
            answer=answer,
            context_used=context_used,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Question processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")

@app.delete("/reset-database/")
async def reset_database():
    """Reset the document database"""
    global qa_system, vector_db
    
    try:
        # Reset global variables
        qa_system = None
        vector_db = None
        
        # Remove database directory
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        
        return {"message": "Database reset successfully"}
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Set environment variables if not already set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY environment variable not set")
        print("Please set it using: set GOOGLE_API_KEY=your_api_key_here")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
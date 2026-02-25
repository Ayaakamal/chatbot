import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging
from sentence_transformers import CrossEncoder
import os
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import io
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- File Processing Functions ---
def process_uploaded_files(uploaded_files):
    """Process uploaded files and extract text"""
    text = ""
    for file in uploaded_files:
        try:
            if file.name.endswith('.pdf'):
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            elif file.name.endswith(('.txt', '.md')):
                text += file.getvalue().decode("utf-8") + "\n"
            elif file.name.endswith('.docx'):
                doc = Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            elif file.name.endswith('.parquet'):
                parquet_file = io.BytesIO(file.getvalue())
                df = pd.read_parquet(parquet_file)
                text += "Parquet File Contents:\n"
                text += df.to_string(index=False) + "\n\n"
                text += f"Data Summary:\n{df.describe().to_string()}\n"
            elif file.name.endswith(('.xlsx', '.xls')):
                excel_file = io.BytesIO(file.getvalue())
                df = pd.read_excel(excel_file)
                text += f"Excel File: {file.name}\n"
                text += df.to_string(index=False) + "\n\n"
                text += f"Data Summary:\n{df.describe().to_string()}\n"
            elif file.name.endswith('.csv'):
                csv_file = io.BytesIO(file.getvalue())
                df = pd.read_csv(csv_file)
                text += f"CSV File: {file.name}\n"
                text += df.to_string(index=False) + "\n\n"
                text += f"Data Summary:\n{df.describe().to_string()}\n"
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            logger.error(f"Error processing {file.name}: {e}", exc_info=True)
    return text

# --- Initialize System with Documents ---
def initialize_qa_system(uploaded_files=None):
    try:
        # Initialize Gemini Pro
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.5)
        embedding_model = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v3")
        
        db = None
        
        # Process documents if provided
        if uploaded_files:
            raw_text = process_uploaded_files(uploaded_files)
            if not raw_text.strip():
                st.error("No text extracted from uploaded files")
                return None, None
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                add_start_index=True
            )
            documents = text_splitter.create_documents([raw_text])
            
            # Create new Chroma instance
            db = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory="./chroma_db"
            )
            st.success("Documents processed and database created successfully!")
            return db, llm
        else:
            # Check for existing Chroma DB
            if os.path.exists("./chroma_db"):
                db = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=embedding_model
                )
                # Verify database has content
                if db._collection.count() == 0:
                    st.error("Database exists but is empty")
                    return None, None
                return db, llm
            else:
                return None, llm

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        st.error(f"System initialization failed: {str(e)}")
        return None, None

def create_qa_chain(db, llm):
    """Create QA chain with chat history support"""
    if not db or not llm:
        return None
        
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,  # Increased from 5 to 8
            "fetch_k": 40,  # Increased from 30 to 40
            "lambda_mult": 0.4  # More similarity-focused
        })
    reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

    # Enhanced prompt with explicit entity tracking
    qa_prompt = PromptTemplate.from_template(
        """You are a helpful AI assistant. Answer the question based on:
        at first detect the language that user write with based on his language answer his question 
        1. The context below
        2. Our conversation history
        3. The current active entity: {active_entity}

        Critical Instructions:
        - ALWAYS resolve "they", "their", "it" to the active entity
        - NEVER say "I don't know" for questions about the active entity
        - If context doesn't contain exact answer, infer from related information
        - Treat different terms for same concept as equivalent (e.g., 'location' and 'address')
        be aware that all questions is about edge pro company , if the user asked any question consider he ask about edge pro , anything he will said
        Conversation History:
        {history}
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        
        
        أنت مساعد ذكاء اصطناعي مفيد. أجب عن السؤال بناءً على:
        1. السياق أدناه
        2. سجل محادثتنا
        3. الكيان النشط الحالي: {active_entity}

        تعليمات حاسمة:
        - دائماً فسّر "هم"، "لهم"، "هو" على أنها تشير إلى الكيان النشط
        - لا تقل أبداً "لا أعرف" للأسئلة المتعلقة بالكيان النشط
        - إذا لم يحتوي السياق على إجابة دقيقة، استنتج من المعلومات ذات الصلة
        - تعامل مع المصطلحات المختلفة للمفهوم نفسه على أنها متكافئة (مثل 'الموقع' و'العنوان')
        - جميع الأسئلة تدور حول شركة EDGE-Pro، إذا سأل المستخدم أي سؤال فاعتبر أنه عن EDGE-Pro مهما كانت صياغته

        سجل المحادثة:
        {history}

        السياق:
        {context}

        السؤال:
        {question}

        الإجابة:
        """
    )

    def enhance_retrieval(question: str, docs: list) -> list:
        try:
            pairs = [(question, doc.page_content) for doc in docs]
            scores = reranker.predict(pairs)
            return [docs[i] for i in sorted(range(len(scores)), 
                    key=lambda k: scores[k], reverse=True)][:4]  # Return top 4
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return docs[:4]

    qa_chain = (
        RunnablePassthrough.assign(
            docs=lambda x: retriever.get_relevant_documents(x["question"]),
        )
        .assign(reranked_docs=lambda x: enhance_retrieval(x["question"], x["docs"]))
        .assign(context=lambda x: "\n".join([d.page_content for d in x["reranked_docs"]]) 
                if x["reranked_docs"] else "No relevant context found.")
        | {
            "context": lambda x: x["context"],
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
            "active_entity": lambda x: x["active_entity"]  # New field
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Chatbot")

# File Upload Section
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Enter Google API Key", type="password")
    st.header(" Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, TXT, DOCX, PARQUET, XLSX, CSV)",
        type=["pdf", "txt", "md", "docx", "parquet", "xlsx", "csv"],
        accept_multiple_files=True
    )
    process_button = st.button("Process Documents")

# System state management
if 'db' not in st.session_state:
    st.session_state.db = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'active_entity' not in st.session_state:
    st.session_state.active_entity = ""

# Initialize LLM (only once)
if st.session_state.llm is None and google_api_key:
    try:
        with st.spinner("Initializing Gemini Pro..."):
            os.environ["GOOGLE_API_KEY"] = google_api_key
            st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        st.stop()

# Process documents when button is clicked
if process_button:
    if not google_api_key:
        st.warning("Please enter your Google API key first")
    elif uploaded_files:
        with st.spinner("Processing documents..."):
            st.session_state.db, _ = initialize_qa_system(uploaded_files)
            if st.session_state.db:
                st.session_state.qa_chain = create_qa_chain(st.session_state.db, st.session_state.llm)
                st.session_state.documents_processed = True
                st.rerun()
    else:
        st.warning("Please upload files first")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I can answer questions about your documents! "
         "Please upload files and enter your Google API key to begin."}
    ]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Process user input
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if not google_api_key:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Please enter your Google API key in the sidebar first."
        })
        st.rerun()
    elif not st.session_state.qa_chain or not st.session_state.documents_processed:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "I need documents to answer questions. Please upload and process documents first."
        })
        st.rerun()
    else:
        try:
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                
                try:
                    # 1. Format conversation history
                    history = "\n".join(
                        [f"{m['role'].capitalize()}: {m['content']}" 
                         for m in st.session_state.messages[:-1]]
                    )
                    
                    # 2. Entity tracking system
                    # Detect new entities in user input
                    if not st.session_state.active_entity:
                        # Try to detect entity from user input
                        entity_detection_prompt = f"""
                        Identify the main entity mentioned in this question: 
                        "{prompt}"
                        
                        Return only the entity name or 'NONE' if no specific entity.
                        """
                        try:
                            entity_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
                            detected_entity = entity_llm.invoke(entity_detection_prompt).content
                            if "NONE" not in detected_entity and detected_entity.strip():
                                st.session_state.active_entity = detected_entity.strip()
                        except:
                            pass
                    
                    # Update entity from assistant responses
                    if st.session_state.messages:
                        last_assistant_responses = [
                            m['content'] for m in st.session_state.messages 
                            if m['role'] == 'assistant'
                        ][-3:]
                        
                        # Extract entities from assistant responses
                        for response in last_assistant_responses:
                            if "EDGE" in response or "Edge" in response:
                                st.session_state.active_entity = "EDGE-Pro"
                    
                    # 3. Create chain input
                    chain_input = {
                        "question": prompt,
                        "history": history,
                        "active_entity": st.session_state.active_entity
                    }
                    
                    # 4. Stream the response
                    for chunk in st.session_state.qa_chain.stream(chain_input):
                        full_response += chunk
                        response_container.markdown(full_response + "▌")
                        
                except Exception as e:
                    logger.error(f"Response generation failed: {e}", exc_info=True)
                    full_response = f"Sorry, I encountered an error: {str(e)}. Please try again."
                
                response_container.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()
        
        except Exception as e:
            logger.error(f"Chat processing failed: {e}", exc_info=True)
            st.error("A serious error occurred. Please refresh the page and try again.")
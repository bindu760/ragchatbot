import streamlit as st
import sqlite3
import hashlib
import json
import os
import tempfile
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import markdown
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from llama_parse import LlamaParse
import fitz  # PyMuPDF as fallback
from groq import Groq  # Import Groq client to fetch available models

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load environment variables with explicit path and override
load_dotenv(override=True)

# Page configuration
st.set_page_config(page_title="Multi-Document RAG Chatbot", layout="wide")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'chunked_data' not in st.session_state:
    st.session_state.chunked_data = []
if 'retrieved_chunks' not in st.session_state:
    st.session_state.retrieved_chunks = []
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'model_fetched' not in st.session_state:
    st.session_state.model_fetched = False

# Load API keys from environment with better error handling
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "").strip()

# Debug: Print API key status (remove in production)
print(f"Groq API Key loaded: {bool(GROQ_API_KEY)} - Length: {len(GROQ_API_KEY)}")
print(f"LlamaParse API Key loaded: {bool(LLAMA_CLOUD_API_KEY)} - Length: {len(LLAMA_CLOUD_API_KEY)}")

# Function to fetch available models from Groq with filtering
def fetch_available_models(api_key):
    """Fetch currently available models from Groq API, filtering out restricted ones"""
    try:
        client = Groq(api_key=api_key)
        models = client.models.list()
        available_models = []
        
        # List of models that typically require special terms or are not for chat
        restricted_patterns = [
            'orpheus',      # Requires terms acceptance
            'canopylabs',   # Requires terms acceptance
            'whisper',      # Audio model
            'embed'         # Embedding model
        ]
        
        for model in models.data:
            model_id = model.id
            # Check if model is restricted
            is_restricted = any(pattern in model_id.lower() for pattern in restricted_patterns)
            
            # Only include non-restricted chat models
            if not is_restricted and not any(x in model_id.lower() for x in ['embed', 'whisper']):
                available_models.append(model_id)
        
        return available_models
    except Exception as e:
        print(f"Error fetching models: {e}")
        return None

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL,
                  question TEXT NOT NULL,
                  answer TEXT NOT NULL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                 (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
             (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user is not None

def save_conversation(username, question, answer):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversations (username, question, answer) VALUES (?, ?, ?)",
             (username, question, answer))
    conn.commit()
    conn.close()

def get_conversation_history(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT question, answer, timestamp FROM conversations WHERE username = ? ORDER BY timestamp DESC",
             (username,))
    history = c.fetchall()
    conn.close()
    return history

# Document processing functions with LlamaParse
def parse_pdf_with_llama(file_content):
    """Parse PDF using LlamaParse"""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        # Initialize LlamaParse with explicit API key
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=True
        )
        
        # Parse the document
        documents = parser.load_data(tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        # Combine all text
        text = "\n\n".join([doc.text for doc in documents])
        return text
    except Exception as e:
        st.warning(f"LlamaParse error: {str(e)}. Falling back to PyMuPDF...")
        return parse_pdf_fallback(file_content)

def parse_pdf_fallback(file_content):
    """Fallback PDF parsing using PyMuPDF"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name
    
    try:
        doc = fitz.open(tmp_file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        os.unlink(tmp_file_path)
        return text
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return None

def parse_markdown(content):
    """Parse markdown content and extract text"""
    html = markdown.markdown(content)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def parse_json(content):
    """Parse JSON content and extract text"""
    try:
        data = json.loads(content)
        return json.dumps(data, indent=2)
    except:
        return content

def process_uploaded_file(uploaded_file):
    """Process uploaded file based on its type"""
    file_content = uploaded_file.getvalue()
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    
    # Get file extension for better detection
    file_extension = os.path.splitext(file_name)[1].lower()
    
    # Check by MIME type first, then by extension
    if file_type == "application/pdf" or file_extension == '.pdf':
        if LLAMA_CLOUD_API_KEY:
            text = parse_pdf_with_llama(file_content)
        else:
            st.warning("LlamaParse API key not found. Using PyMuPDF fallback...")
            text = parse_pdf_fallback(file_content)
    elif file_type in ["application/json", "text/json"] or file_extension == '.json':
        text = parse_json(file_content.decode('utf-8'))
    elif file_type in ["text/markdown", "text/x-markdown"] or file_extension in ['.md', '.markdown']:
        text = parse_markdown(file_content.decode('utf-8'))
    elif file_type == "text/plain" or file_extension == '.txt':
        # Handle text files
        try:
            text = file_content.decode('utf-8')
            if not text.strip():
                st.warning(f"File {file_name} appears to be empty.")
                return None
            return text
        except UnicodeDecodeError:
            try:
                text = file_content.decode('latin-1')
                return text
            except:
                st.error(f"Could not decode {file_name} as text.")
                return None
    else:
        # Try to parse as text as a last resort
        try:
            text = file_content.decode('utf-8')
            if text.strip():
                return text
            else:
                st.error(f"Unsupported file type: {file_type} for {file_name}")
                return None
        except:
            st.error(f"Unsupported file type: {file_type} for {file_name}")
            return None
    
    return text

def chunk_documents(texts: List[str], chunk_size: int = 500, chunk_overlap: int = 50):
    """Chunk documents into smaller pieces"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = []
    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        for j, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"source": f"doc_{i}_chunk_{j}", "chunk_index": j}
            ))
    
    return documents

def create_vector_store(documents):
    """Create FAISS vector store from documents"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# RAG functions
def get_retrieved_chunks(query: str, vector_store, k: int = 5):
    """Retrieve relevant chunks for a query"""
    if vector_store is None:
        return []
    
    docs = vector_store.similarity_search(query, k=k)
    return docs

def generate_answer(query: str, context_docs: List[Document], groq_api_key: str, model_name: str):
    """Generate answer using Groq LLM with specified model"""
    if not groq_api_key:
        return "Groq API key not found in environment variables. Please check your .env file."
    
    # Prepare context
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Truncate context if it's too long (roughly 2000 words / 8000 tokens)
    # This is a safety measure to prevent token limit errors
    words = context.split()
    if len(words) > 2000:  # Approximate limit for 8000 token context
        context = " ".join(words[:2000]) + "\n\n[Context truncated due to length...]"
    
    # Escape any curly braces in the context to prevent f-string parsing errors
    context_escaped = context.replace('{', '{{').replace('}', '}}')
    
    # Create prompt with escaped context
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context. "
                   "Use only the information from the context to answer. "
                   "If you don't know the answer, say so. "
                   "Keep your answers concise and focused."),
        ("human", f"Context: {context_escaped}\n\nQuestion: {query}\n\nAnswer:")
    ])
    
    try:
        llm = ChatGroq(
            api_key=groq_api_key,
            model=model_name,
            temperature=0.3,
            streaming=True,
            max_tokens=1000  # Limit response length to save tokens
        )
        
        chain = prompt | llm
        response = chain.invoke({})
        return response.content
    except Exception as e:
        # If still too long, try with fewer chunks
        if "length" in str(e).lower() and len(context_docs) > 3:
            st.warning("Context too long. Trying with fewer chunks...")
            # Try with half the chunks
            half_chunks = context_docs[:len(context_docs)//2]
            return generate_answer(query, half_chunks, groq_api_key, model_name)
        return f"Error generating answer: {str(e)}"

# Main UI
def login_page():
    st.title("🔐 Login / Register")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username", key="register_username")
            new_password = st.text_input("Choose Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            submit = st.form_submit_button("Register")
            
            if submit:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif create_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

def main_app():
    st.title(f"🤖 Multi-Document RAG Chatbot")
    st.sidebar.write(f"Welcome, **{st.session_state.username}**!")
    
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
    
    # Display API status - SECURE VERSION (no key preview)
    st.sidebar.subheader("🔑 API Configuration")
    
    # Groq API Status - Only shows if key is loaded, never displays the key
    if GROQ_API_KEY:
        st.sidebar.success("✅ Groq API Key loaded")
    else:
        st.sidebar.error("❌ Groq API Key not found in .env")
    
    # Fetch and display available models (only once)
    if GROQ_API_KEY and not st.session_state.model_fetched:
        with st.sidebar.spinner("Fetching available models..."):
            models = fetch_available_models(GROQ_API_KEY)
            if models and len(models) > 0:
                st.session_state.available_models = models
                st.session_state.model_fetched = True
                st.sidebar.success(f"✅ Found {len(models)} available models")
            else:
                st.sidebar.error("❌ Could not fetch models or no available models")
                # Fallback to known working models
                fallback_models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
                st.session_state.available_models = fallback_models
                st.session_state.model_fetched = True
                st.sidebar.warning("Using fallback model list (these models may be deprecated)")
    
    # Display available models in sidebar
    if st.session_state.available_models:
        st.sidebar.subheader("🤖 Available Models")
        # Show only the first 5 models to keep UI clean
        for model in st.session_state.available_models[:5]:
            st.sidebar.text(f"• {model}")
        if len(st.session_state.available_models) > 5:
            st.sidebar.text(f"... and {len(st.session_state.available_models) - 5} more")
    else:
        st.sidebar.warning("No models available. Please check your Groq API key.")
    
    # Model selection
    selected_model = None
    if st.session_state.available_models:
        selected_model = st.sidebar.selectbox(
            "Select Model",
            st.session_state.available_models,
            index=0,
            key="model_selector"
        )
    
    # LlamaParse API Status - Only shows if key is loaded
    st.sidebar.subheader("📄 Document Parser")
    if LLAMA_CLOUD_API_KEY:
        st.sidebar.success("✅ LlamaParse API Key loaded")
    else:
        st.sidebar.error("❌ LlamaParse API Key not found in .env")
        st.sidebar.info("ℹ️ PDF parsing will use PyMuPDF fallback")
    
    if TAVILY_API_KEY:
        st.sidebar.success("✅ Tavily API Key loaded")
    else:
        st.sidebar.info("ℹ️ Tavily API Key not configured (optional)")
    
    # Document upload section
    st.header("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload JSON, Markdown, PDF, or Text files",
        type=['json', 'md', 'pdf', 'txt'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=500,
            key="chunk_size_input"
        )
    with col2:
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=50,
            key="chunk_overlap_input"
        )
    
    if uploaded_files and st.button("Process Documents", key="process_docs_button"):
        with st.spinner("Processing documents..."):
            processed_texts = []
            failed_files = []
            
            for uploaded_file in uploaded_files:
                text = process_uploaded_file(uploaded_file)
                if text:
                    processed_texts.append(text)
                    st.session_state.documents.append({
                        "name": uploaded_file.name,
                        "content": text[:500] + "..." if len(text) > 500 else text
                    })
                else:
                    failed_files.append(uploaded_file.name)
            
            if processed_texts:
                # Chunk documents
                documents = chunk_documents(processed_texts, chunk_size, chunk_overlap)
                st.session_state.chunked_data = [
                    {"chunk_index": i, "content": doc.page_content}
                    for i, doc in enumerate(documents)
                ]
                
                # Create vector store
                st.session_state.vector_store = create_vector_store(documents)
                
                st.success(f"✅ Processed {len(processed_texts)} documents successfully!")
                if failed_files:
                    st.warning(f"⚠️ Failed to process: {', '.join(failed_files)}")
            else:
                st.error("No documents were successfully processed")
    
    # Display chunked data
    if st.session_state.chunked_data:
        with st.expander("📊 View Chunked Data"):
            chunk_df = pd.DataFrame(st.session_state.chunked_data)
            st.dataframe(chunk_df, height=300)
    
    # Question answering section
    st.header("❓ Ask Questions")
    
    # Query input with unique key
    query = st.text_input(
        "Enter your question:",
        placeholder="Type your question here...",
        key="query_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        k_retrieval = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            key="k_retrieval_slider"
        )
    
    if query and st.button("Get Answer", key="get_answer_button"):
        if st.session_state.vector_store is None:
            st.warning("Please upload and process some documents first.")
        elif not GROQ_API_KEY:
            st.warning("Groq API key not found. Please check your .env file.")
        elif not selected_model:
            st.warning("No model selected. Please wait for models to load.")
        else:
            with st.spinner(f"Generating answer using {selected_model}..."):
                # Retrieve relevant chunks
                retrieved_docs = get_retrieved_chunks(query, st.session_state.vector_store, k_retrieval)
                st.session_state.retrieved_chunks = [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in retrieved_docs
                ]
                
                # Generate answer
                answer = generate_answer(query, retrieved_docs, GROQ_API_KEY, selected_model)
                
                # Display answer
                st.subheader("💬 Answer")
                st.write(answer)
                
                # Save conversation
                save_conversation(st.session_state.username, query, answer)
                st.session_state.conversation_history.append({
                    "question": query,
                    "answer": answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
    
    # Display retrieved chunks
    if st.session_state.retrieved_chunks:
        with st.expander("🔍 View Retrieved Chunks"):
            for i, chunk in enumerate(st.session_state.retrieved_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.text(chunk['content'])
                st.text(f"Metadata: {chunk['metadata']}")
                st.divider()
    
    # Conversation history
    st.header("📝 Conversation History")
    
    # Load history from database
    if st.session_state.username:
        history = get_conversation_history(st.session_state.username)
        if history:
            for idx, (question, answer, timestamp) in enumerate(history[:10]):  # Show last 10
                with st.expander(f"📌 {timestamp}", key=f"history_expander_{idx}"):
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")
        else:
            st.info("No conversation history yet.")

# Initialize database
init_db()

# Main app flow
if st.session_state.authenticated:
    main_app()
else:
    login_page()
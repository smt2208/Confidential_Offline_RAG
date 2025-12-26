# Confidential Offline RAG System

A robust **Retrieval-Augmented Generation (RAG)** system for confidential document interrogation records, designed to run completely offline using local LLMs via Ollama. The system combines advanced RAG techniques with conversational AI to provide accurate, context-aware responses from your private document collection.

## 🎯 Overview

This project implements an end-to-end RAG pipeline that processes confidential PDF documents, stores them in a vector database, and provides an interactive chat interface for querying the documents using natural language. All processing happens locally, ensuring complete data privacy.

## 🏗️ Architecture & Flow

### Project Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    1. DATA INGESTION PHASE                          │
│  PDF Documents → PyPDF Loader → Parent Document Strategy            │
│  → Text Chunking → Embeddings → ChromaDB Vector Store               │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    2. QUERY PROCESSING PHASE                        │
│                                                                     │
│  User Query → [LangGraph Workflow]                                  │
│                                                                     │
│  Node 1: Query Reformulation                                        │
│  ├─ Analyze conversation history                                    │
│  └─ Reformulate query for better context (if needed)                │
│                                                                     │
│  Node 2: Document Retrieval                                         │
│  ├─ Vector similarity search in ChromaDB                            │
│  ├─ Retrieve top-k parent documents                                 │
│  └─ Build context from retrieved documents                          │
│                                                                     │
│  Node 3: Response Generation                                        │
│  ├─ Combine context + conversation history                          │
│  ├─ Generate response using Ollama LLM                              │
│  └─ Update conversation memory (SQLite checkpointer)                │
│                                                                     │
│  → AI Response                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    3. USER INTERACTION                              │
│  Streamlit Frontend ↔ FastAPI Backend ↔ RAG Pipeline               │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

**Backend Architecture:**
- **RAG_pipeline.py**: Main orchestrator that initializes and coordinates all RAG components
- **src/embeddings.py**: Handles text embeddings via Ollama (mxbai-embed-large)
- **src/vectorstore.py**: Manages ChromaDB vector database operations
- **src/retriever.py**: Implements parent document retrieval strategy
- **src/llm.py**: Manages Ollama LLM for generation (llama3.2:3b)
- **src/workflow.py**: LangGraph workflow with 3 nodes (reformulate → retrieve → generate)
- **api.py**: FastAPI REST API for chat interactions
- **config.py**: Centralized configuration management

**Frontend:**
- **app.py**: Streamlit-based chat interface with conversation persistence

**Data Pipeline:**
- **notebooks/Data_injestion.ipynb**: Jupyter notebook for PDF indexing and vector database creation

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.10+**
- **LangChain 0.3.0+** - RAG orchestration framework
- **LangGraph 0.2.0+** - Workflow state management
- **Ollama** - Local LLM inference engine
- **ChromaDB 0.5.23+** - Vector database
- **FastAPI 0.115.0+** - Backend REST API
- **Streamlit 1.41.0+** - Frontend UI
- **SQLite** - Conversation checkpointing

### LLM Models
- **Generation**: llama3.2:3b
- **Embeddings**: mxbai-embed-large:335m

### Document Processing
- **PyMuPDF** - PDF text extraction
- **LangChain Text Splitters** - Chunking strategy

### Additional Tools
- **Pydantic 2.10+** - Data validation
- **httpx** - Async HTTP client
- **python-dotenv** - Environment configuration

## 📦 Setup for Local Use

### Prerequisites

1. **Install Ollama** ([https://ollama.ai](https://ollama.ai))
   - Download and install for your OS

2. **Pull Required Models**
   ```bash
   ollama pull llama3.2:3b
   ollama pull mxbai-embed-large:335m
   ```

### Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Confidential_Offline_RAG
   ```

2. **Create Conda Environment** (Recommended)
   
   **Prerequisites:**
   - Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   
   ```bash
   # Create new conda environment with Python 3.11.1
   conda create -n crag python=3.11.1 -y
   
   # Activate the environment
   conda activate crag
   ```

3. **Alternative: Create Virtual Environment**
   
   If you prefer venv instead of conda:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment (Optional)**
   
   Create a `.env` file in the project root to override defaults:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2:3b
   OLLAMA_EMBEDDING_MODEL=mxbai-embed-large:335m
   ANONYMIZED_TELEMETRY=False
   ```

### Data Preparation

1. **Add Your PDF Documents**
   - Place PDF files in `Data/Data/` directory

2. **Index Documents**
   - Open `notebooks/Data_injestion.ipynb` in Jupyter
   - Run all cells to process PDFs and create vector database
   - This creates `backend/chroma_db/` with indexed documents

### Running the Application

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

This will start:
- Ollama service
- FastAPI backend (http://localhost:8000)
- Streamlit frontend (http://localhost:8501)

### Access Points

- **Chat Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 🐛 Debug Tools

The project includes comprehensive debugging capabilities for tracing RAG pipeline execution.

### RAG Pipeline Tracer

Located in `debug_tools/`, this tool provides step-by-step execution traces:

**Features:**
- Query reformulation analysis
- Document retrieval details with source tracking
- Context building visualization
- Full prompt construction preview
- LLM invocation tracking
- Conversation memory updates
- Interactive mode for testing multiple queries

**Usage:**

**Windows:**
```bash
cd debug_tools
run_debug.bat
```

**Linux/Mac:**
```bash
cd debug_tools
chmod +x run_debug.sh
./run_debug.sh
```

**Direct Python:**
```bash
python debug_tools/trace_rag_pipeline.py
```

See `debug_tools/README.md` for detailed usage instructions.

## 📊 Project Structure

```
Confidential_Offline_RAG/
├── backend/
│   ├── src/                    # Core RAG components
│   │   ├── embeddings.py       # Embedding management
│   │   ├── vectorstore.py      # ChromaDB interface
│   │   ├── retriever.py        # Retrieval logic
│   │   ├── llm.py              # LLM management
│   │   └── workflow.py         # LangGraph workflow
│   ├── models/                 # Pydantic models
│   ├── prompts/                # System prompts
│   ├── api.py                  # FastAPI application
│   ├── RAG_pipeline.py         # Main orchestrator
│   ├── config.py               # Configuration
│   ├── logger.py               # Logging setup
│   ├── chroma_db/              # Vector database (generated)
│   ├── parent_docs/            # Parent document storage (generated)
│   └── checkpoints/            # Conversation state (generated)
├── frontend/
│   └── app.py                  # Streamlit UI
├── debug_tools/
│   ├── trace_rag_pipeline.py   # Debug tracer
│   ├── run_debug.bat           # Windows runner
│   └── run_debug.sh            # Linux/Mac runner
├── notebooks/
│   └── Data_injestion.ipynb    # Document indexing
├── Data/
│   └── Data/                   # PDF documents (add yours here)
├── requirements.txt            # Python dependencies
├── start.bat                   # Windows launcher
└── start.sh                    # Linux/Mac launcher
```

## 🖼️ Sample Output Screenshots

### Query 1

![path/to/screenshot1.png](<assets/sample outputs/Screenshot (765).png>)

### Follow-up Question
![path/to/screenshot2.png](<assets/sample outputs/Screenshot (766).png>)

### Out of context question

![path/to/screenshot3.png](<assets/sample outputs/Screenshot (767).png>)

### Speific question about a document

![path/to/screenshot4.png](<assets/sample outputs/Screenshot (768).png>)

## 🔧 Configuration

Key configuration parameters in `backend/config.py`:

- **CHUNK_SIZE**: 400 characters (chunk size for embeddings)
- **CHUNK_OVERLAP**: 50 characters (overlap between chunks)
- **TOP_K_PARENT_DOCS**: 2 (number of documents to retrieve)
- **LLM_TEMPERATURE**: 0.2 (response creativity)
- **SEARCH_TYPE**: "similarity" (vector search method)

> **📋 Note**: This project works best for records or small documents. For larger documents, increase the `CHUNK_SIZE` in `backend/config.py` but keep it under 2000 characters to maintain optimal retrieval performance and context quality.

## 📝 How It Works

1. **Indexing Phase** (One-time setup)
   - PDFs are loaded and merged into parent documents (1 per file)
   - Documents are split into smaller chunks for embedding
   - Both chunks and parent docs are stored in ChromaDB
   - Each chunk maintains a reference to its parent document

2. **Query Phase** (Runtime)
   - User submits a question via Streamlit UI
   - Query is reformulated based on conversation history (if needed)
   - Vector similarity search retrieves relevant chunks
   - Full parent documents are fetched and combined as context
   - LLM generates response using context + conversation history
   - Response is returned and conversation state is saved

3. **Conversation Continuity**
   - SQLite-based checkpointer stores conversation threads
   - Each user session has a unique thread ID
   - Follow-up questions leverage previous context

## 🚀 Advanced Features

- **Parent Document Retrieval**: Retrieves full documents for better context, not just chunks
- **Query Reformulation**: Automatically improves queries based on chat history
- **Conversation Memory**: Maintains context across multiple interactions
- **Offline Operation**: Complete privacy - no data leaves your machine
- **Structured Output**: Uses Pydantic models for type-safe LLM responses
- **Async API**: FastAPI with proper lifecycle management
- **Debug Tracing**: Comprehensive pipeline visibility for troubleshooting

## 📄 License

See [LICENSE](LICENSE) file for details.

---

**Note**: This is a local-first, privacy-focused RAG system. All document processing and AI inference happens on your machine using Ollama be sure you have a capable GPU.
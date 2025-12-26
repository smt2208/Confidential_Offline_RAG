# RAG Pipeline Debug Tools

This folder contains debugging and tracing tools for the RAG pipeline.

## Tools

### 1. RAG Pipeline Tracer (`trace_rag_pipeline.py`)

A comprehensive debugging tool that traces the entire RAG pipeline execution from query to response.

**Features:**
- Step-by-step execution trace
- Query reformulation analysis
- Document retrieval details
- Context building visualization
- Prompt construction preview
- LLM invocation tracking
- Conversation memory management
- Interactive mode for multiple queries

**What it shows:**
1. **Query Reformulation**: Shows if and how the query is reformulated based on conversation history
2. **Document Retrieval**: Displays retrieved parent documents, their sources, and content previews
3. **Context Building**: Shows how multiple documents are combined into context
4. **Prompt Construction**: Previews the full prompt sent to the LLM
5. **LLM Invocation**: Tracks the request and response from the language model
6. **Memory Update**: Shows how conversation history is maintained

## Usage

### Windows
```bash
cd debug_tools
run_debug.bat
```

### Linux/Mac
```bash
cd debug_tools
chmod +x run_debug.sh
./run_debug.sh
```

### Direct Python
```bash
python debug_tools/trace_rag_pipeline.py
```

## Interactive Mode

The tracer runs in interactive mode where you can:
1. Enter a query
2. See the complete pipeline execution trace
3. Enter another query (conversation history is maintained)
4. Type `exit`, `quit`, or `q` to stop

## Example Output

```
================================================================================
RAG PIPELINE DEBUG TRACER
================================================================================
Timestamp: 2025-12-26 10:30:45
Model: llama3.2:3b
Embedding Model: mxbai-embed-large:335m
Chunk Size: 800, Overlap: 100
Top K Documents: 5
================================================================================

[STEP 0] Initializing RAG Components...
✓ All components initialized

================================================================================
NEW QUERY SIMULATION
================================================================================
User Query: 'Who is Arjun Chowdhury?'
================================================================================

[STEP 1] QUERY REFORMULATION
...
[STEP 2] DOCUMENT RETRIEVAL
...
[STEP 3] CONTEXT BUILDING
...
[STEP 4] PROMPT CONSTRUCTION
...
[STEP 5] LLM INVOCATION
...
[STEP 6] MEMORY UPDATE
...
[FINAL RESPONSE]
...
```

## Configuration

The tracer uses the same configuration as the main application from `backend/config.py`:
- OLLAMA_MODEL
- OLLAMA_EMBEDDING_MODEL
- CHUNK_SIZE, CHUNK_OVERLAP
- TOP_K_PARENT_DOCS
- LLM_TEMPERATURE

## Requirements

Requires the same dependencies as the main RAG application. Ensure:
- Ollama is running
- Vector database is indexed
- All Python packages are installed

## Troubleshooting

If you encounter errors:
1. Ensure Ollama is running: `ollama serve`
2. Check that the vector database exists (run data ingestion notebook)
3. Verify all configuration paths in `backend/config.py`

#!/bin/bash

echo "========================================"
echo "Confidential Offline RAG System"
echo "========================================"
echo ""

# Start Ollama in background
echo "Starting Ollama service..."
ollama serve &
sleep 3

# Start FastAPI backend
echo "Starting FastAPI backend..."
python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000 &
sleep 2

# Start Streamlit frontend
echo "Starting Streamlit frontend..."
streamlit run frontend/app.py &

echo ""
echo "========================================"
echo "Application started!"
echo ""
echo "Frontend: http://localhost:8501"
echo "API Docs: http://localhost:8000/docs"
echo "========================================"

wait

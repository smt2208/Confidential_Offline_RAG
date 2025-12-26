@echo off
echo ========================================
echo Confidential Offline RAG System
echo ========================================
echo.

echo Starting Ollama service...
start /B ollama serve

timeout /t 3 /nobreak > nul

echo Starting FastAPI backend...
start cmd /k "cd /d %~dp0 && python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000"

timeout /t 2 /nobreak > nul

echo Starting Streamlit frontend...
start cmd /k "cd /d %~dp0 && streamlit run frontend/app.py"

echo.
echo ========================================
echo Application started!
echo.
echo Frontend: http://localhost:8501
echo API Docs: http://localhost:8000/docs
echo ========================================

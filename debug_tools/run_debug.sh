#!/bin/bash

echo "========================================"
echo "RAG Pipeline Debug Tracer"
echo "========================================"
echo ""
echo "This tool traces the entire RAG pipeline execution:"
echo "- Query reformulation"
echo "- Document retrieval"
echo "- Context building"
echo "- Prompt construction"
echo "- LLM invocation"
echo "- Memory management"
echo ""
echo "========================================"
echo ""

cd "$(dirname "$0")/.."
python debug_tools/trace_rag_pipeline.py

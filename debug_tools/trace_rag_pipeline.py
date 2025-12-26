"""
RAG Pipeline Debug Tracer

This script simulates and traces the entire RAG pipeline execution,
showing each step from query input to final response generation.

USAGE:
    python trace_rag_pipeline.py

The tracer will:
1. Initialize all RAG components (embeddings, vectorstore, retriever, LLM)
2. Provide an interactive prompt for testing queries
3. Show detailed step-by-step execution of the RAG pipeline
4. Display query reformulation, document retrieval, context building, and response generation

This is useful for:
- Debugging RAG pipeline issues
- Understanding how queries are processed
- Testing retrieval effectiveness
- Validating LLM responses
"""

import sys
import os
from datetime import datetime
from typing import Any, cast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import config
from backend.src.embeddings import EmbeddingsManager
from backend.src.vectorstore import VectorStoreManager
from backend.src.retriever import RetrieverManager
from backend.src.llm import LLMManager
from backend.src.workflow import RAGWorkflow
from backend.prompts.prompts import SYSTEM_PROMPT


class RAGPipelineDebugger:
    """
    Debug and trace RAG pipeline execution.

    This class simulates the complete RAG workflow by manually stepping through
    each stage of the pipeline, providing detailed logging and output for debugging.

    Unlike the production LangGraph workflow, this debugger:
    - Uses direct component calls instead of graph orchestration
    - Maintains its own conversation history
    - Provides verbose step-by-step output
    - Shows full prompts and intermediate results
    """

    def __init__(self):
        """Initialize the debug tracer with all RAG components."""
        print("=" * 80)
        print("RAG PIPELINE DEBUG TRACER")
        print("=" * 80)
        print(f"Timestamp: {datetime.now()}")
        print(f"Model: {config.OLLAMA_MODEL}")
        print(f"Embedding Model: {config.OLLAMA_EMBEDDING_MODEL}")
        print(f"Chunk Size: {config.CHUNK_SIZE}, Overlap: {config.CHUNK_OVERLAP}")
        print(f"Top K Documents: {config.TOP_K_PARENT_DOCS}")
        print("=" * 80)

        print("\n[STEP 0] Initializing RAG Components...")
        self.embeddings = EmbeddingsManager()  # Text embedding service
        self.vectorstore = VectorStoreManager(self.embeddings)  # ChromaDB vector storage
        self.retriever = RetrieverManager(self.vectorstore, search_type=config.SEARCH_TYPE)  # Document retrieval
        self.llm = LLMManager()  # Ollama LLM service
        self.workflow = RAGWorkflow(self.llm, self.retriever)  # LangGraph workflow (for reference)
        print("✓ All components initialized\n")

        self.conversation_history = []  # Manual conversation tracking for debugging
    
    def trace_query_reformulation(self, user_query: str, messages: list) -> str:
        """Trace query reformulation step."""
        print("\n" + "=" * 80)
        print("[STEP 1] QUERY REFORMULATION")
        print("=" * 80)
        print(f"Original Query: '{user_query}'")
        print(f"Conversation History Length: {len(messages)} messages")
        
        if len(messages) < 2:
            print("→ Early conversation detected - using original query")
            reformulated = str(user_query)
            was_reformulated = False
        else:
            print(f"→ Analyzing last {min(6, len(messages))} messages for context")
            
            recent_history = messages[-6:]
            history_text = "\n".join([
                f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content[:100]}..."
                for msg in recent_history
            ])
            
            print("\nConversation Context:")
            print("-" * 80)
            print(history_text)
            print("-" * 80)
            
            reformulation_prompt = f"""Given the conversation history, analyze the current query and determine if it needs reformulation to be standalone and self-contained.

            CHAT HISTORY:
            {history_text}

            CURRENT USER QUERY: "{user_query}"

            INSTRUCTIONS:
            - If the query contains pronouns (he/she/it/they) or vague references (this/that/these/those), replace them with specific names/terms from the chat history
            - If the query specifies a particular person by name, only provide information from records that match that exact name. Do not alter the query to include information of someone else as it may lead to incorrect retrieval or the query asked was not relevent or the person does not exists in the records.
            - If the query is already clear and standalone, return it unchanged
            - Focus on making the query suitable for document retrieval by adding necessary context
            - Do not hallucinate or add information not present in the chat history"""
            
            print("\nSending reformulation request to LLM...")
            
            from backend.src.workflow import Reformulate
            structured_llm = self.llm.get_llm().with_structured_output(Reformulate)
            result = cast(Reformulate, structured_llm.invoke(reformulation_prompt))
            reformulated = str(result.reformulated_query)
            was_reformulated = result.was_reformulated
        
        print(f"\nReformulated Query: '{reformulated}'")
        print(f"Was Reformulated: {was_reformulated}")
        
        return str(reformulated)
    
    def trace_retrieval(self, query: str) -> list:
        """Trace document retrieval step."""
        print("\n" + "=" * 80)
        print("[STEP 2] DOCUMENT RETRIEVAL")
        print("=" * 80)
        print(f"Search Query: '{query}'")
        print(f"Search Type: {config.SEARCH_TYPE}")
        print(f"Target Parent Documents: {config.TOP_K_PARENT_DOCS}")
        
        print("\n→ Searching vector store for relevant chunks...")
        results = self.retriever.retrieve(query, k=config.TOP_K_PARENT_DOCS)
        
        print(f"\n✓ Retrieved {len(results)} parent document(s)")
        
        if results:
            print("\nRetrieved Documents:")
            print("-" * 80)
            for idx, doc in enumerate(results, 1):
                source = doc.metadata.get('source', 'Unknown')
                filename = os.path.basename(source) if os.path.sep in source or '/' in source else source
                total_pages = doc.metadata.get('total_pages', 'Unknown')
                content_length = len(doc.page_content)
                
                print(f"\nDocument {idx}:")
                print(f"  Source: {filename}")
                print(f"  Total Pages: {total_pages}")
                print(f"  Content Length: {content_length} characters")
                print(f"  Preview: {doc.page_content[:200]}...")
        else:
            print("⚠ No documents found!")
        
        return results
    
    def trace_context_building(self, results: list) -> str:
        """Trace context building from retrieved documents."""
        print("\n" + "=" * 80)
        print("[STEP 3] CONTEXT BUILDING")
        print("=" * 80)
        
        if not results:
            print("→ No documents to build context from")
            return ""
        
        print(f"→ Building context from {len(results)} document(s)")
        
        contexts = []
        for idx, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(source) if os.path.sep in source or '/' in source else source
            
            context_part = f"""SOURCE FILE: {filename}
=== CONTENT ===
{doc.page_content}
=== END ===
"""
            contexts.append(context_part)
            print(f"  ✓ Added document {idx}: {filename}")
        
        full_context = "\n".join(contexts)
        print(f"\n✓ Total context length: {len(full_context)} characters")
        
        return full_context
    
    def trace_prompt_construction(self, query: str, context: str, messages: list) -> str:
        """Trace prompt construction for LLM."""
        print("\n" + "=" * 80)
        print("[STEP 4] PROMPT CONSTRUCTION")
        print("=" * 80)
        
        history_text = "\n".join([
            f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
            for msg in messages
        ])
        
        print(f"Conversation History: {len(history_text)} characters")
        print(f"Context Length: {len(context)} characters")
        print(f"User Query: '{query}'")
        
        prompt = SYSTEM_PROMPT.format(
            context=context,
            history_of_conversation=history_text,
            query=query
        )
        
        # print(f"\n✓ Full prompt constructed: {len(prompt)} characters")
        
        print("\nFull Prompt Preview:")
        print("-" * 80)
        print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
        print("-" * 80)
        
        return prompt
    
    def trace_llm_invocation(self, prompt: str) -> str:
        """Trace LLM invocation."""
        print("\n" + "=" * 80)
        print("[STEP 5] LLM INVOCATION")
        print("=" * 80)
        print(f"Model: {config.OLLAMA_MODEL}")
        print(f"Temperature: {config.LLM_TEMPERATURE}")
        print(f"Prompt Length: {len(prompt)} characters")
        
        print("\n→ Sending request to LLM...")
        try:
            response = self.llm.get_llm().invoke(prompt)
            
            # Ensure response_text is always a string
            if isinstance(response.content, str):
                response_text = response.content
            elif isinstance(response.content, list):
                response_text = str(response.content)
            else:
                response_text = str(response.content) if response.content else ""
                
        except Exception as e:
            print(f"\n❌ LLM invocation failed: {e}")
            response_text = f"Error: {e}"
        
        print(f"\n✓ Response received: {len(response_text)} characters")
        
        return str(response_text)
    
    def trace_memory_update(self, user_query: str, response: str):
        """Trace conversation memory update."""
        print("\n" + "=" * 80)
        print("[STEP 6] MEMORY UPDATE")
        print("=" * 80)
        
        from langchain_core.messages import HumanMessage, AIMessage
        
        user_msg = HumanMessage(content=user_query)
        ai_msg = AIMessage(content=response)
        
        self.conversation_history.append(user_msg)
        self.conversation_history.append(ai_msg)
        
        print(f"→ Added user message to history")
        print(f"→ Added AI response to history")
        print(f"✓ Total messages in history: {len(self.conversation_history)}")
    
    def simulate_query(self, user_query: str):
        """
        Simulate a complete query through the RAG pipeline.

        This method manually steps through each stage of the RAG pipeline,
        calling each tracing method in sequence to provide detailed debugging output.

        Args:
            user_query: The user's input query to process
        """
        print("\n\n" + "=" * 80)
        print(f"NEW QUERY SIMULATION")
        print("=" * 80)
        print(f"User Query: '{user_query}'")
        print("=" * 80)

        # Step 1: Query reformulation (handle pronouns, context from conversation)
        reformulated_query = self.trace_query_reformulation(user_query, self.conversation_history)
        print(f"Reformulated Query: '{reformulated_query}'")

        # Step 2: Document retrieval (find relevant documents from vector store)
        retrieved_docs = self.trace_retrieval(reformulated_query)

        # Step 3: Context building (format retrieved documents for LLM)
        context = self.trace_context_building(retrieved_docs)

        # Step 4: Prompt construction (build final prompt with system instructions)
        prompt = self.trace_prompt_construction(reformulated_query, context, self.conversation_history)

        # Step 5: LLM invocation (generate response)
        response = self.trace_llm_invocation(prompt)

        # Step 6: Memory update (add to conversation history)
        self.trace_memory_update(user_query, response)

        print("\n" + "=" * 80)
        print("[FINAL RESPONSE]")
        print("=" * 80)
        print(response)
        print("=" * 80)

        return response


def main():
    """
    Main entry point for debug tracer.

    Provides an interactive command-line interface for testing RAG queries.
    Users can enter queries and see detailed step-by-step pipeline execution.
    """
    debugger = RAGPipelineDebugger()

    print("\n\n" + "=" * 80)
    print("INTERACTIVE DEBUG MODE")
    print("=" * 80)
    print("Enter queries to trace through the pipeline.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 80)

    while True:
        try:
            user_input = input("\n\nYour Query: ").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nExiting debug tracer...")
                break

            if not user_input:
                continue

            debugger.simulate_query(user_input)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

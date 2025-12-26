"""LangGraph workflow for RAG system."""

import os
from typing import TypedDict, Annotated, cast
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from backend.src.llm import LLMManager
from backend.src.retriever import RetrieverManager
from backend.prompts.prompts import SYSTEM_PROMPT
from backend import config
from backend.logger import get_logger

log = get_logger(__name__)


class ChatState(TypedDict):
    """
    Chat conversation state for LangGraph workflow.

    This TypedDict defines the structure of the state that flows through
    the RAG workflow nodes. It maintains conversation context and intermediate results.
    """
    messages: Annotated[list[BaseMessage], add_messages]  # Conversation history with automatic merging
    context: str  # Retrieved document context for the current query
    user_query: str  # Original user query
    reformulated_query: str  # Query after reformulation (if needed)


class Reformulate(BaseModel):
    """
    Structured output for query reformulation.

    Used with Ollama's structured output to ensure consistent reformulation results.
    """
    reformulated_query: str = Field(description="Reformulated query string.")
    was_reformulated: bool = Field(description="True if the query was reformulated, False if the original query was returned unchanged.")


class RAGWorkflow:
    """
    Manages the RAG workflow using LangGraph.

    This class defines the three main nodes of the RAG pipeline:
    1. Query reformulation based on conversation history
    2. Document retrieval and context building
    3. Response generation using retrieved context
    """

    def __init__(self, llm_manager: LLMManager, retriever_manager: RetrieverManager):
        """
        Initialize the RAG workflow with LLM and retriever components.

        Args:
            llm_manager: Configured LLM manager for text generation
            retriever_manager: Configured retriever for document search
        """
        self.llm = llm_manager.get_llm()
        self.retriever = retriever_manager

    def reformulate_query(self, state: ChatState) -> dict:
        """
        Reformulate the query for better retrieval based on conversation history.

        This node analyzes the conversation context and may rephrase the user's query
        to make it more suitable for document retrieval (e.g., replacing pronouns with
        specific names from the chat history).

        Args:
            state: Current chat state containing user query and message history

        Returns:
            dict: Updated state with reformulated_query field
        """
        user_query = state["user_query"]
        messages = state.get("messages", [])

        # Skip reformulation for early conversation (not enough context)
        if len(messages) < 2:
            log.info("First Message - using query as-is")
            return {"reformulated_query": user_query}

        # Get recent history for context (last 6 messages to avoid token limits)
        recent_history = messages[-6:]
        history_text = self._format_history(recent_history)
        
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

        try:
            # Use structured output to get Reformulate object directly from LLM
            structured_llm = self.llm.with_structured_output(Reformulate)
            result = cast(Reformulate, structured_llm.invoke(reformulation_prompt))

            # Determine if reformulated by comparing queries (more reliable than LLM flag)
            was_reformulated = result.reformulated_query.strip() != user_query.strip()

            log.info(f"Query: '{user_query}' → '{result.reformulated_query}' ({'reformulated' if was_reformulated else 'unchanged'})")
            return {"reformulated_query": result.reformulated_query}
        except Exception as e:
            log.error(f"Reformulation error: {e}")
            return {"reformulated_query": user_query}

    def retrieve_context(self, state: ChatState) -> dict:
        """
        Retrieve relevant documents and build context for the query.

        This node performs the RAG retrieval step, finding the most relevant
        documents from the vector store and formatting them into a context string
        that will be used for answer generation.

        Args:
            state: Current chat state with reformulated query

        Returns:
            dict: Updated state with context field containing retrieved documents
        """
        query = state.get("reformulated_query", state["user_query"])
        results = self.retriever.retrieve(query, k=config.TOP_K_PARENT_DOCS)

        if results:
            # Combine multiple documents into formatted context
            contexts = []
            for idx, doc in enumerate(results, 1):
                source = doc.metadata.get('source', 'Unknown')
                # Extract filename from path if it's a path
                filename = os.path.basename(source) if os.path.sep in source or '/' in source else source

                context_part = f"""SOURCE FILE: {filename}
                === CONTENT ===
                {doc.page_content}
                === END ===
                """
                contexts.append(context_part)
                log.info(f"Retrieved {idx}: {filename}")

            return {"context": "\n".join(contexts)}
        else:
            log.warning(f"No documents found for: {query[:50]}")
            return {"context": ""}

    def generate_response(self, state: ChatState) -> dict:
        """
        Generate the final response using retrieved context and conversation history.

        This node constructs the final prompt with system instructions, context,
        and conversation history, then generates the AI response.

        Args:
            state: Current chat state with context and user query

        Returns:
            dict: Updated state with new AI message added to conversation
        """
        # Format conversation history for the prompt
        history_text = self._format_history(state.get("messages", []))

        # Construct the final prompt using the system template
        query_to_use = state.get("reformulated_query", state["user_query"])
        prompt = SYSTEM_PROMPT.format(
                context=state["context"],
                history_of_conversation=history_text,
                query=query_to_use
            )

        # Create user message for conversation tracking
        user_msg = HumanMessage(content=state["user_query"])

        # Generate response using the LLM
        response = self.llm.invoke(prompt)
        log.info(f"Generated response for: {query_to_use[:50]}")

        # Return both user and AI messages to update conversation state
        return {"messages": [user_msg, response]}

    def _format_history(self, messages):
        """
        Format message history for inclusion in prompts.

        Converts LangChain message objects into a simple text format
        suitable for inclusion in LLM prompts.

        Args:
            messages: List of BaseMessage objects

        Returns:
            str: Formatted conversation history
        """
        formatted = []
        for msg in messages:
            role = "User" if msg.type == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    def build_graph(self, checkpointer):
        """
        Build and compile the LangGraph workflow.

        Creates the state graph with three nodes (reformulate, retrieve, generate)
        and defines the execution flow. The graph maintains conversation state
        and supports persistence through the checkpointer.

        Args:
            checkpointer: LangGraph checkpointer for conversation persistence

        Returns:
            CompiledStateGraph: The executable workflow graph
        """
        graph = StateGraph(ChatState)

        # Add workflow nodes (functions that process the state)
        graph.add_node("reformulate", self.reformulate_query)  # Query preprocessing
        graph.add_node("retrieve", self.retrieve_context)      # Document retrieval
        graph.add_node("generate", self.generate_response)     # Response generation

        # Define execution flow: START -> reformulate -> retrieve -> generate -> END
        graph.add_edge(START, "reformulate")
        graph.add_edge("reformulate", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        # Compile the graph with checkpointer for conversation persistence
        return graph.compile(checkpointer=checkpointer)

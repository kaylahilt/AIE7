"""LangGraph server configuration for LangGraph Studio."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.agent_graph_with_helpfulness import build_agent_graph_with_helpfulness

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatOpenAI(
    model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    openai_api_base=os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1'),
    temperature=0,
)

# System instructions
SYSTEM_INSTRUCTION = (
    'You are a helpful AI assistant with access to various tools including web search, '
    'academic paper search, and document retrieval. '
    'Use the appropriate tools to answer user questions accurately and thoroughly. '
    'If you cannot find relevant information using the available tools, '
    'clearly state that you were unable to find the requested information.'
)

FORMAT_INSTRUCTION = (
    'Set response status to input_required if the user needs to provide more information to complete the request.'
    'Set response status to error if there is an error while processing the request.'
    'Set response status to completed if the request is complete.'
)

# Build and export the graph (no checkpointer - LangGraph API handles persistence)
graph = build_agent_graph_with_helpfulness(
    model,
    SYSTEM_INSTRUCTION,
    FORMAT_INSTRUCTION,
    checkpointer=None
)

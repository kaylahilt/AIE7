"""LangGraph Agent Library

A library for LangGraph agents with caching, monitoring, and agent integration.
"""

from .agents import create_langgraph_agent
from .caching import CacheBackedEmbeddings, setup_llm_cache
from .rag import ProductionRAGChain
from .models import get_openai_model
from .guarded_agents import create_guarded_langgraph_agent, GuardrailsTestSuite, setup_default_guards

__version__ = "0.1.0"
__all__ = [
    "create_langgraph_agent",
    "create_guarded_langgraph_agent",
    "CacheBackedEmbeddings",
    "setup_llm_cache",
    "ProductionRAGChain",
    "get_openai_model",
    "GuardrailsTestSuite",
    "setup_default_guards",
]


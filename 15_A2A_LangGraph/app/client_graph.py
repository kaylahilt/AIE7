"""Simple LangGraph client that uses the A2A agent as a tool."""

import os
from typing import Annotated, List, Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.a2a_client_tool import A2AAgentTool

# Load environment variables
load_dotenv()


class ClientState(dict):
    """State for the client graph."""
    messages: Annotated[List[BaseMessage], add_messages]


def create_client_agent():
    """Create a simple agent that can delegate to the A2A agent."""
    
    # Initialize the model
    model = ChatOpenAI(
        model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1'),
        temperature=0,
    )
    
    # Create the A2A agent tool
    a2a_tool = A2AAgentTool()
    tools = [a2a_tool]
    
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: ClientState) -> Dict[str, Any]:
        """Call the model with available tools."""
        messages = state["messages"]
        
        # Add system message for context
        system_message = HumanMessage(content="""You are a helpful assistant that can delegate complex queries to a specialized A2A agent.

The A2A agent has access to:
- Web search for current information
- Academic paper search on arXiv  
- Document retrieval from loaded PDFs

When a user asks a question that would benefit from these capabilities, use the a2a_agent tool to get a comprehensive answer.
For simple questions that don't require these tools, you can answer directly.""")
        
        # Prepend system message if not already present
        if not messages or messages[0].content != system_message.content:
            messages = [system_message] + messages
            
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: ClientState) -> str:
        """Determine if we should continue to tool execution or end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    # Build the graph
    workflow = StateGraph(ClientState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


# Export the compiled graph
client_graph = create_client_agent()

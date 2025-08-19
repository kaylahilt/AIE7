"""A2A Client Tool for making calls to the A2A agent from another LangGraph."""

import asyncio
import logging
from typing import Any, Dict
from uuid import uuid4

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH


class A2AQueryInput(BaseModel):
    """Input schema for A2A agent queries."""
    query: str = Field(description="The question or request to send to the A2A agent")


class A2AAgentTool(BaseTool):
    """Tool for making API calls to the A2A agent."""
    
    name: str = "a2a_agent"
    description: str = """
    Call the A2A agent to get answers using web search, academic paper search, and document retrieval.
    Use this tool when you need to:
    - Search the web for current information
    - Find academic papers on arXiv
    - Retrieve information from documents
    The A2A agent has access to powerful tools and can provide comprehensive answers.
    """
    args_schema: type[BaseModel] = A2AQueryInput
    base_url: str = Field(default="http://localhost:10000", description="Base URL for the A2A agent")
    
    def __init__(self, base_url: str = "http://localhost:10000", **kwargs):
        super().__init__(base_url=base_url, **kwargs)
    
    def _run(self, query: str) -> str:
        """Synchronous wrapper for async A2A call."""
        return asyncio.run(self._arun(query))
    
    async def _arun(self, query: str) -> str:
        """Make an async call to the A2A agent."""
        logger = logging.getLogger(__name__)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
                # Initialize A2A card resolver
                resolver = A2ACardResolver(
                    httpx_client=httpx_client,
                    base_url=self.base_url,
                )
                
                # Fetch the agent card
                logger.info(f"Fetching agent card from {self.base_url}")
                agent_card = await resolver.get_agent_card()
                
                # Initialize A2A client
                client = A2AClient(
                    httpx_client=httpx_client, 
                    agent_card=agent_card
                )
                
                # Prepare the message
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [
                            {'kind': 'text', 'text': query}
                        ],
                        'message_id': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                # Send the message and get response
                logger.info(f"Sending query to A2A agent: {query}")
                response = await client.send_message(request)
                
                # Extract the text response from the result
                if response.root and response.root.result:
                    result = response.root.result
                    
                    # Try to get from artifacts
                    if hasattr(result, 'artifacts') and result.artifacts:
                        for artifact in result.artifacts:
                            if hasattr(artifact, 'parts') and artifact.parts:
                                for part in artifact.parts:
                                    if hasattr(part, 'text') and part.text:
                                        return part.text
                                    # Also try accessing as dict
                                    elif isinstance(part, dict) and 'text' in part:
                                        return part['text']
                
                # Fallback: try to parse from the response dump
                response_dict = response.model_dump()
                if 'result' in response_dict and 'artifacts' in response_dict['result']:
                    artifacts = response_dict['result']['artifacts']
                    for artifact in artifacts:
                        if 'parts' in artifact:
                            for part in artifact['parts']:
                                if 'text' in part and part['text']:
                                    return part['text']
                
                return "No response received from A2A agent"
                
        except Exception as e:
            logger.error(f"Error calling A2A agent: {e}")
            return f"Error communicating with A2A agent: {str(e)}"

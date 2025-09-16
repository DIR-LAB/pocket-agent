import pytest
import logging
import asyncio
from unittest.mock import Mock, AsyncMock
from litellm import Router
from fastmcp import FastMCP
from pocket_agent.agent import AgentConfig, PocketAgent
from pocket_agent.client import PocketAgentClient
from pocket_agent.agent import PocketAgent, StepResult
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
from mcp.types import CallToolRequestParams


ROUTER_CONFIG = {
    "models": [
        {
            "model_name": "gpt-5",
            "litellm_params": {
                "model": "gpt-5",
                "tpm": 3000000,
                "rpm": 5000
            }
        }
    ]
}



@pytest.fixture
def mock_router():
    """Mock LiteLLM Router for testing"""
    router = Mock(spec=Router)
    router.acompletion = AsyncMock()
    return router


@pytest.fixture
def fastmcp_server():
    """Fixture that creates a FastMCP server with tools, resources, and prompts."""
    server = FastMCP(name="TestServer",
                     instructions="This server provides tools to greet people and add numbers")

    # Add a tool
    @server.tool(
        description="Greet someone by name"
    )
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    # Add a second tool
    @server.tool(
        description="Add two numbers together"
    )
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @server.tool(
        description="Sleep for a given number of seconds"
    )
    async def sleep(seconds: float) -> str:
        """Sleep for a given number of seconds."""
        await asyncio.sleep(seconds)
        return f"Slept for {seconds} seconds"

    # Add a resource
    @server.resource(uri="data://users",
                     description="Get a list of users")
    async def get_users():
        return ["Alice", "Bob", "Charlie"]

    # Add a resource template
    @server.resource(uri="data://user/{user_id}",
                     description="Get a user by ID")
    async def get_user(user_id: str):
        return {"id": user_id, "name": f"User {user_id}", "active": True}

    # Add a prompt
    @server.prompt(
        description="Welcome message"
    )
    def welcome(name: str) -> str:
        """Example greeting prompt."""
        return f"Welcome to FastMCP, {name}!"

    return server





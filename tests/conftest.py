import pytest
import logging
import asyncio
from unittest.mock import Mock, AsyncMock
from litellm import Router
from fastmcp import FastMCP
from pocket_agent.agent import AgentConfig, MCPAgent
from pocket_agent.client import Client

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
def sample_agent_config():
    """Basic agent configuration for testing"""
    return AgentConfig(
        llm_model="gpt-5",
        agent_id="0000",
        context_id="1111",
        system_prompt="You are a testing assistant.",
        messages=[],
        allow_images=False,
        completion_kwargs={"tool_choice": "auto"}
    )



@pytest.fixture
def mock_router():
    """Mock LiteLLM Router for testing"""
    router = Mock(spec=Router)
    router.acompletion = AsyncMock()
    return router


@pytest.fixture
def mock_mcp_server_config():
    """Mock MCP server configuration"""
    return {
        "server_command": ["python", "-m", "test_server"],
        "server_args": []
    }


@pytest.fixture
def fastmcp_server():
    """Fixture that creates a FastMCP server with tools, resources, and prompts."""
    server = FastMCP("TestServer")

    # Add a tool
    @server.tool
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    # Add a second tool
    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @server.tool
    async def sleep(seconds: float) -> str:
        """Sleep for a given number of seconds."""
        await asyncio.sleep(seconds)
        return f"Slept for {seconds} seconds"

    # Add a resource
    @server.resource(uri="data://users")
    async def get_users():
        return ["Alice", "Bob", "Charlie"]

    # Add a resource template
    @server.resource(uri="data://user/{user_id}")
    async def get_user(user_id: str):
        return {"id": user_id, "name": f"User {user_id}", "active": True}

    # Add a prompt
    @server.prompt
    def welcome(name: str) -> str:
        """Example greeting prompt."""
        return f"Welcome to FastMCP, {name}!"

    return server



@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    return Mock(spec=logging.Logger)


@pytest.fixture
def mock_client():
    """Mock MCP client for testing"""
    client = Mock(spec=Client)
    client._get_openai_tools = AsyncMock(return_value=[])
    client.call_tools = AsyncMock(return_value=[])
    return client


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for testing"""
    mock_message = Mock()
    mock_message.content = "Test response content"
    mock_message.tool_calls = None
    mock_message.model_dump.return_value = {
        "role": "assistant",
        "content": "Test response content"
    }
    
    mock_choice = Mock()
    mock_choice.message = mock_message
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    
    return mock_response

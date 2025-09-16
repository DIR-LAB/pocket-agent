import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock
from pocket_agent.client import PocketAgentClient, ToolResult
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
from mcp.types import CallToolRequestParams


class TestToolResult:
    """Test the ToolResult dataclass"""
    
    def test_tool_result_creation(self):
        """Test ToolResult creation with all fields"""
        result = ToolResult(
            tool_call_id="call_123",
            tool_call_name="test_tool",
            tool_result_content=[{"type": "text", "text": "result"}]
        )
        
        assert result.tool_call_id == "call_123"
        assert result.tool_call_name == "test_tool"
        assert result.tool_result_content == [{"type": "text", "text": "result"}]

    def test_tool_result_with_extra_data(self):
        """Test ToolResult with optional _extra field"""
        result = ToolResult(
            tool_call_id="call_123",
            tool_call_name="test_tool",
            tool_result_content=[{"type": "text", "text": "result"}],
            _extra={"metadata": "value"}
        )
        
        assert result._extra == {"metadata": "value"}


class TestPocketAgentClient:
    """Simplified tests for PocketAgentClient using real FastMCP servers"""

    @pytest.fixture
    def real_mcp_config(self):
        """MCP configuration that points to real test server"""
        return {
            "mcpServers": {
                "test_server": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["server.py"],
                    "cwd": os.path.dirname(__file__)
                }
            }
        }

    def test_client_initialization(self, real_mcp_config):
        """Test basic client initialization"""
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        # Basic checks
        assert client.mcp_server_config == real_mcp_config
        assert client.client_logger.name == "pocket_agent.client"
        assert hasattr(client, 'client')

    def test_client_with_custom_loggers(self, real_mcp_config):
        """Test client initialization with custom loggers"""
        import logging
        custom_logger = logging.getLogger("custom.test")
        
        client = PocketAgentClient(
            mcp_config=real_mcp_config,
            client_logger=custom_logger
        )
        
        assert client.client_logger == custom_logger

    @pytest.mark.asyncio
    async def test_get_tools_mcp_format(self, real_mcp_config):
        """Test getting tools in MCP format from real server"""
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        tools = await client.get_tools(format="mcp")
        
        # Should have tools from our test server
        assert len(tools) >= 3
        tool_names = {tool.name for tool in tools}
        assert "greet" in tool_names
        assert "add" in tool_names
        assert "sleep" in tool_names

    @pytest.mark.asyncio
    async def test_get_tools_openai_format(self, real_mcp_config):
        """Test getting tools in OpenAI format from real server"""
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        tools = await client.get_tools(format="openai")
        
        # Should have tools converted to OpenAI format
        assert len(tools) >= 3
        
        # Check OpenAI format structure
        greet_tool = next((t for t in tools if t["function"]["name"] == "greet"), None)
        assert greet_tool is not None
        assert greet_tool["type"] == "function"
        assert "parameters" in greet_tool["function"]

    @pytest.mark.asyncio
    async def test_call_tool_greet(self, real_mcp_config):
        """Test calling the greet tool on real server"""
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        # Create a proper tool call request
        tool_call = CallToolRequestParams(
            name="greet", 
            arguments={"name": "Test User"}
        )
        tool_call.id = "test_call_123"
        
        result = await client.call_tool(tool_call)
        
        assert isinstance(result, ToolResult)
        assert result.tool_call_id == "test_call_123"
        assert result.tool_call_name == "greet"
        assert len(result.tool_result_content) > 0
        assert "Hello, Test User!" in result.tool_result_content[0]["text"]

    @pytest.mark.asyncio
    async def test_call_tool_add(self, real_mcp_config):
        """Test calling the add tool on real server"""
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        tool_call = CallToolRequestParams(
            name="add",
            arguments={"a": 15, "b": 25}
        )
        tool_call.id = "test_add_456"
        
        result = await client.call_tool(tool_call)
        
        assert result.tool_call_id == "test_add_456"
        assert result.tool_call_name == "add"
        assert "40" in result.tool_result_content[0]["text"]


    @pytest.mark.asyncio
    async def test_get_tool_input_format(self, real_mcp_config):
        """Test getting tool input format from real server"""
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        # Get the format for the greet tool
        format_info = await client.get_tool_input_format("greet")
        
        # Should return schema information
        assert format_info is not None
        # The exact format depends on the server implementation

    @pytest.mark.asyncio
    async def test_get_tool_input_format_missing_tool(self, real_mcp_config):
        """Test getting format for non-existent tool"""
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        with pytest.raises(ValueError, match="Tool nonexistent_tool not found"):
            await client.get_tool_input_format("nonexistent_tool")


    @pytest.mark.asyncio
    async def test_fastmcp_direct_comparison(self, fastmcp_server):
        """Compare PocketAgentClient behavior with direct FastMCP usage"""
        # Direct FastMCP client
        fastmcp_client = Client(transport=FastMCPTransport(fastmcp_server))
        
        async with fastmcp_client:
            # Get tools directly from FastMCP
            fastmcp_tools = await fastmcp_client.list_tools()
            fastmcp_tool_names = {t.name for t in fastmcp_tools}
            
            # Call tool directly
            fastmcp_result = await fastmcp_client.call_tool("greet", {"name": "Direct"})
            
            assert "greet" in fastmcp_tool_names
            assert "add" in fastmcp_tool_names
            assert "Hello, Direct!" in str(fastmcp_result.content)

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, real_mcp_config):
        """Test calling async tools (like sleep)"""
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        # Call the async sleep tool with short duration
        sleep_call = CallToolRequestParams(
            name="sleep",
            arguments={"seconds": 0.1}  # Short sleep for testing
        )
        sleep_call.id = "sleep_test"
        
        import time
        start_time = time.time()
        result = await client.call_tool(sleep_call)
        end_time = time.time()
        
        assert result.tool_call_id == "sleep_test"
        assert "Slept for 0.1 seconds" in result.tool_result_content[0]["text"]
        # Should have actually waited
        assert (end_time - start_time) >= 0.1

    def test_client_configuration_handling(self):
        """Test client handles different configuration formats"""
        # Test with minimal config
        minimal_config = {
            "mcpServers": {
                "test": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["server.py"]
                }
            }
        }
        
        client = PocketAgentClient(mcp_config=minimal_config)
        assert client.mcp_server_config == minimal_config

    @pytest.mark.asyncio
    async def test_transform_tool_call_request(self, real_mcp_config):
        """Test tool call request transformation"""
        from litellm.types.utils import ChatCompletionMessageToolCall, Function
        
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        # Create OpenAI-style tool call
        openai_tool_call = ChatCompletionMessageToolCall(
            function=Function(
                arguments='{"name":"Transform Test"}',
                name="greet"
            ),
            id="transform_test",
            type="function"
        )
        
        # Transform it
        mcp_call = client.transform_tool_call_request(openai_tool_call)
        
        assert mcp_call.name == "greet"
        assert mcp_call.arguments == {"name": "Transform Test"}
        assert mcp_call.id == "transform_test"

import pytest
import asyncio
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
from pocket_agent.client import PocketAgentClient
from pocket_agent.agent import PocketAgent, AgentConfig
from mcp.types import CallToolRequestParams


class TestFastMCPServerIntegration:
    """Tests using the MCP server functionality with pocket_agent.client.Client"""

    @pytest.mark.asyncio
    async def test_client_get_tools_from_fastmcp_server(self, fastmcp_server):
        """Test that the client can retrieve tools from FastMCP server"""
        # Use FastMCP's in-memory transport - no external processes needed!
        fastmcp_client = Client(transport=FastMCPTransport(fastmcp_server))
        
        async with fastmcp_client:
            tools = await fastmcp_client.list_tools()
            
            assert len(tools) == 3
            tool_names = {tool.name for tool in tools}
            assert tool_names == {"greet", "add", "sleep"}

    @pytest.mark.asyncio
    async def test_client_call_tools_with_fastmcp_server(self, fastmcp_server):
        """Test calling tools on the actual FastMCP server"""
        fastmcp_client = Client(transport=FastMCPTransport(fastmcp_server))
        
        async with fastmcp_client:
            # Test calling the greet tool
            result = await fastmcp_client.call_tool("greet", {"name": "Alice"})
            assert "Hello, Alice!" in str(result.content)
            
            # Test calling the add tool  
            result = await fastmcp_client.call_tool("add", {"a": 5, "b": 3})
            assert "8" in str(result.content)


    @pytest.mark.asyncio
    async def test_fastmcp_server_tools_directly(self, fastmcp_server):
        """Test the FastMCP server tools directly"""
        client = Client(transport=FastMCPTransport(fastmcp_server))
        
        async with client:
            # List available tools
            tools = await client.list_tools()
            assert len(tools) == 3
            
            # Test each tool
            greet_result = await client.call_tool("greet", {"name": "World"})
            assert "Hello, World!" in str(greet_result.content)
            
            add_result = await client.call_tool("add", {"a": 10, "b": 20})
            assert "30" in str(add_result.content)
            
            # Test async tool
            sleep_result = await client.call_tool("sleep", {"seconds": 0.1})
            assert "Slept for 0.1 seconds" in str(sleep_result.content)

    @pytest.mark.asyncio
    async def test_tool_error_handling_with_fastmcp(self, fastmcp_server):
        """Test error handling with real FastMCP server"""
        client = Client(transport=FastMCPTransport(fastmcp_server))
        
        async with client:
            # Test calling tool with wrong parameters
            with pytest.raises(Exception):  # FastMCP will raise real errors
                await client.call_tool("greet", {"wrong_param": "test"})

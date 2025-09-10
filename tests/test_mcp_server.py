import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pocket_agent.agent import MCPAgent, AgentConfig
from pocket_agent.client import Client, ToolResult
from litellm import Router


class TestFastMCPServerIntegration:
    """Tests using the MCP server functionality with pocket_agent.client.Client"""

    @pytest.mark.asyncio
    async def test_client_get_tools_from_mcp_server(self, mock_mcp_server_config):
        """Test that the client can retrieve tools from MCP server"""
        with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
            # Mock the MCP client instance
            mock_client_instance = Mock()
            
            # Mock tools that match our FastMCP server fixture - fix the name attribute
            mock_greet_tool = Mock()
            mock_greet_tool.name = "greet"
            mock_greet_tool.inputSchema = {"type": "object", "properties": {"name": {"type": "string"}}}
            
            mock_add_tool = Mock()
            mock_add_tool.name = "add"
            mock_add_tool.inputSchema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}
            
            mock_sleep_tool = Mock()
            mock_sleep_tool.name = "sleep"
            mock_sleep_tool.inputSchema = {"type": "object", "properties": {"seconds": {"type": "number"}}}
            
            mock_tools = [mock_greet_tool, mock_add_tool, mock_sleep_tool]
            
            mock_client_instance.get_tools = AsyncMock(return_value=mock_tools)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_mcp_client.return_value = mock_client_instance
            
            client = Client(mock_mcp_server_config)
            tools = await client._get_mcp_tools()
            
            assert len(tools) == 3
            tool_names = [tool.name for tool in tools]
            assert "greet" in tool_names
            assert "add" in tool_names
            assert "sleep" in tool_names

    @pytest.mark.asyncio
    async def test_client_get_openai_tools(self, mock_mcp_server_config):
        """Test converting MCP tools to OpenAI format"""
        with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
            mock_client_instance = Mock()
            
            # Mock MCP tool with proper name attribute
            mock_mcp_tool = Mock()
            mock_mcp_tool.name = "greet"
            mock_client_instance.get_tools = AsyncMock(return_value=[mock_mcp_tool])
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_mcp_client.return_value = mock_client_instance
            
            # Mock the transform function
            mock_openai_tool = {
                "type": "function",
                "function": {
                    "name": "greet",
                    "description": "Greet someone by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]
                    }
                }
            }
            
            with patch('pocket_agent.client.transform_mcp_tool_to_openai_tool', return_value=mock_openai_tool):
                client = Client(mock_mcp_server_config)
                openai_tools = await client._get_openai_tools()
                
                assert len(openai_tools) == 1
                assert openai_tools[0]["function"]["name"] == "greet"
                assert openai_tools[0]["type"] == "function"

    @pytest.mark.asyncio
    async def test_client_call_greet_tool(self, mock_mcp_server_config):
        """Test calling the greet tool through the client"""
        with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
            mock_client_instance = Mock()
            
            # Mock successful tool call result
            mock_mcp_result = Mock()
            mock_mcp_result.structuredContent = "Hello, Alice!"
            mock_client_instance.call_tool = AsyncMock(return_value=mock_mcp_result)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_mcp_client.return_value = mock_client_instance
            
            client = Client(mock_mcp_server_config)
            
            # Create mock tool call
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.name = "greet"
            mock_tool_call.model_dump.return_value = {
                "id": "call_123",
                "function": {"name": "greet", "arguments": '{"name": "Alice"}'}
            }
            
            with patch('pocket_agent.client.transform_openai_tool_call_request_to_mcp_tool_call_request') as mock_transform:
                mock_mcp_request = Mock()
                mock_mcp_request.arguments = {"name": "Alice"}
                mock_transform.return_value = mock_mcp_request
                
                result = await client.call_tool(mock_tool_call)
                
                assert isinstance(result, ToolResult)
                assert result.tool_call_id == "call_123"
                assert result.tool_call_name == "greet"
                assert result.tool_result_content == [{"type": "text", "text": "Hello, Alice!"}]
                
                # Verify the underlying MCP client was called correctly
                mock_client_instance.call_tool.assert_called_once_with("greet", {"name": "Alice"})

    @pytest.mark.asyncio
    async def test_client_call_add_tool(self, mock_mcp_server_config):
        """Test calling the add tool through the client"""
        with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
            mock_client_instance = Mock()
            
            # Mock add tool result
            mock_mcp_result = Mock()
            mock_mcp_result.structuredContent = "7"  # 3 + 4 = 7
            mock_client_instance.call_tool = AsyncMock(return_value=mock_mcp_result)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_mcp_client.return_value = mock_client_instance
            
            client = Client(mock_mcp_server_config)
            
            # Create mock tool call
            mock_tool_call = Mock()
            mock_tool_call.id = "call_456"
            mock_tool_call.name = "add"
            mock_tool_call.model_dump.return_value = {
                "id": "call_456",
                "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'}
            }
            
            with patch('pocket_agent.client.transform_openai_tool_call_request_to_mcp_tool_call_request') as mock_transform:
                mock_mcp_request = Mock()
                mock_mcp_request.arguments = {"a": 3, "b": 4}
                mock_transform.return_value = mock_mcp_request
                
                result = await client.call_tool(mock_tool_call)
                
                assert result.tool_call_id == "call_456"
                assert result.tool_call_name == "add"
                assert result.tool_result_content == [{"type": "text", "text": "7"}]
                mock_client_instance.call_tool.assert_called_once_with("add", {"a": 3, "b": 4})

    @pytest.mark.asyncio
    async def test_client_call_multiple_tools_parallel(self, mock_mcp_server_config):
        """Test calling multiple tools in parallel"""
        with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
            mock_client_instance = Mock()
            
            # Mock results for both tools
            mock_greet_result = Mock()
            mock_greet_result.structuredContent = "Hello, Bob!"
            
            mock_add_result = Mock() 
            mock_add_result.structuredContent = "15"
            
            # Setup call_tool to return different results based on tool name
            async def mock_call_tool(tool_name, args):
                if tool_name == "greet":
                    return mock_greet_result
                elif tool_name == "add":
                    return mock_add_result
                else:
                    # Return a default mock result for any other tool names
                    default_result = Mock()
                    default_result.structuredContent = f"Result for {tool_name}"
                    return default_result
                    
            mock_client_instance.call_tool = AsyncMock(side_effect=mock_call_tool)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_mcp_client.return_value = mock_client_instance
            
            client = Client(mock_mcp_server_config)
            
            # Create multiple tool calls with proper name attributes
            mock_tool_call_1 = Mock()
            mock_tool_call_1.id = "call_1"
            mock_tool_call_1.name = "greet"
            mock_tool_call_1.model_dump = Mock(return_value={
                "id": "call_1", "function": {"name": "greet", "arguments": '{"name": "Bob"}'}})
                
            mock_tool_call_2 = Mock()
            mock_tool_call_2.id = "call_2" 
            mock_tool_call_2.name = "add"
            mock_tool_call_2.model_dump = Mock(return_value={
                "id": "call_2", "function": {"name": "add", "arguments": '{"a": 7, "b": 8}'}})
                
            mock_tool_calls = [mock_tool_call_1, mock_tool_call_2]
            
            with patch('pocket_agent.client.transform_openai_tool_call_request_to_mcp_tool_call_request') as mock_transform:
                def mock_transform_func(openai_tool):
                    if openai_tool["function"]["name"] == "greet":
                        result = Mock()
                        result.arguments = {"name": "Bob"}
                        return result
                    else:
                        result = Mock()
                        result.arguments = {"a": 7, "b": 8}
                        return result
                        
                mock_transform.side_effect = mock_transform_func
                
                results = await client.call_tools(mock_tool_calls)
                
                assert len(results) == 2
                assert results[0].tool_call_name == "greet"
                assert results[0].tool_result_content == [{"type": "text", "text": "Hello, Bob!"}]
                assert results[1].tool_call_name == "add" 
                assert results[1].tool_result_content == [{"type": "text", "text": "15"}]

    @pytest.mark.asyncio
    async def test_agent_with_mcp_server_tools(self, mock_mcp_server_config, mock_router):
        """Test complete agent workflow with MCP server tools"""
        # Setup agent config
        config = AgentConfig(
            llm_model="gpt-4",
            system_prompt="You are a helpful assistant that can greet people and do math.",
            allow_images=False
        )
        
        # Mock the client
        mock_client = Mock()
        mock_client._get_openai_tools = AsyncMock(return_value=[
            {
                "type": "function",
                "function": {
                    "name": "greet",
                    "description": "Greet someone by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "add",
                    "description": "Add two numbers together.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ])
        
        # Mock tool execution results
        mock_greet_result = ToolResult(
            tool_call_id="call_greet",
            tool_call_name="greet",
            tool_result_content=[{"type": "text", "text": "Hello, Bob!"}]
        )
        
        mock_add_result = ToolResult(
            tool_call_id="call_add",
            tool_call_name="add",
            tool_result_content=[{"type": "text", "text": "15"}]
        )
        
        mock_client.call_tools = AsyncMock(return_value=[mock_greet_result, mock_add_result])
        
        with patch.object(MCPAgent, '_init_client', return_value=mock_client):
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=config
            )
            
            # Create proper mock tool call objects (not just dictionaries)
            mock_tool_call_1 = Mock()
            mock_tool_call_1.id = "call_greet"
            mock_tool_call_1.function = Mock()
            mock_tool_call_1.function.name = "greet"
            mock_tool_call_1.function.arguments = '{"name": "Bob"}'
            mock_tool_call_1.type = "function"
            # Add the dict representation for when the agent processes it
            mock_tool_call_1.__dict__.update({
                "id": "call_greet",
                "function": {"name": "greet", "arguments": '{"name": "Bob"}'},
                "type": "function"
            })
            
            mock_tool_call_2 = Mock()
            mock_tool_call_2.id = "call_add"
            mock_tool_call_2.function = Mock()
            mock_tool_call_2.function.name = "add"
            mock_tool_call_2.function.arguments = '{"a": 7, "b": 8}'
            mock_tool_call_2.type = "function"
            mock_tool_call_2.__dict__.update({
                "id": "call_add",
                "function": {"name": "add", "arguments": '{"a": 7, "b": 8}'},
                "type": "function"
            })
            
            mock_tool_calls = [mock_tool_call_1, mock_tool_call_2]
            
            # Mock LLM message with tool calls
            mock_message = Mock()
            mock_message.content = None
            mock_message.tool_calls = mock_tool_calls
            mock_message.model_dump.return_value = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_greet",
                        "function": {"name": "greet", "arguments": '{"name": "Bob"}'},
                        "type": "function"
                    },
                    {
                        "id": "call_add", 
                        "function": {"name": "add", "arguments": '{"a": 7, "b": 8}'},
                        "type": "function"
                    }
                ]
            }
            
            mock_choice = Mock()
            mock_choice.message = mock_message
            
            mock_llm_response = Mock()
            mock_llm_response.choices = [mock_choice]
            
            mock_router.acompletion = AsyncMock(return_value=mock_llm_response)
            
            # Add user message and use step() instead of generate()
            agent.add_user_message("Please greet Bob and add 7 and 8")
            
            # Use step() which handles the complete workflow including tool execution
            await agent.step()
            
            # Verify the workflow
            mock_client._get_openai_tools.assert_called()
            mock_client.call_tools.assert_called_once()
            
            # Verify the conversation flow - should have user message, assistant message, and tool result messages
            assert len(agent.messages) >= 4  # user + assistant + 2 tool results
            
            # Check that tool result messages were added
            tool_messages = [msg for msg in agent.messages if msg.get("role") == "tool"]
            assert len(tool_messages) == 2

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_mcp_server_config):
        """Test error handling when MCP tool fails"""
        with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
            mock_client_instance = Mock()
            
            # Mock tool error
            from fastmcp.exceptions import ToolError
            mock_client_instance.call_tool = AsyncMock(side_effect=ToolError("unexpected_keyword_argument test"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_mcp_client.return_value = mock_client_instance
            
            client = Client(mock_mcp_server_config)
            
            # Mock get_tools for error handling - fix the tool name attribute
            mock_tool = Mock()
            mock_tool.name = "greet"
            mock_tool.inputSchema = '{"name": "string"}'
            mock_client_instance.get_tools = AsyncMock(return_value=[mock_tool])
            
            # Create mock tool call
            mock_tool_call = Mock()
            mock_tool_call.id = "call_error"
            mock_tool_call.name = "greet"
            mock_tool_call.model_dump.return_value = {
                "id": "call_error",
                "function": {"name": "greet", "arguments": '{"wrong_param": "test"}'}
            }
            
            with patch('pocket_agent.client.transform_openai_tool_call_request_to_mcp_tool_call_request') as mock_transform:
                mock_mcp_request = Mock()
                mock_mcp_request.arguments = {"wrong_param": "test"}
                mock_transform.return_value = mock_mcp_request
                
                result = await client.call_tool(mock_tool_call)
                
                # Should return error message instead of raising
                assert isinstance(result, ToolResult)
                assert "unexpected keyword argument" in result.tool_result_content[0]["text"]
                assert "expected format" in result.tool_result_content[0]["text"]

    @pytest.mark.asyncio
    async def test_parse_tool_result_with_content_array(self, mock_mcp_server_config):
        """Test parsing tool result with mixed content types"""
        with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
            mock_client_instance = Mock()
            mock_mcp_client.return_value = mock_client_instance
            
            client = Client(mock_mcp_server_config)
            
            # Create mock result with content array
            mock_text_content = Mock()
            mock_text_content.type = "text"
            mock_text_content.text = "Text content"
            
            mock_image_content = Mock()
            mock_image_content.type = "image"
            mock_image_content.imageBase64 = "base64data"
            
            mock_result = Mock()
            mock_result.structuredContent = None
            mock_result.content = [mock_text_content, mock_image_content]
            
            parsed = client._parse_tool_result(mock_result)
            
            assert len(parsed) == 2
            assert parsed[0] == {"type": "text", "text": "Text content"}
            assert parsed[1] == {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,base64data"}
            }

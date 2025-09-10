import pytest
import logging
from unittest.mock import Mock, AsyncMock, patch
from pocket_agent.client import Client, ToolResult
from fastmcp.client.logging import LogMessage
from fastmcp.exceptions import ToolError


class TestToolResult:
    def test_tool_result_creation(self):
        """Test ToolResult dataclass"""
        result = ToolResult(
            tool_call_id="call_123",
            tool_call_name="test_tool",
            tool_result_content=[{"type": "text", "text": "result"}]
        )
        
        assert result.tool_call_id == "call_123"
        assert result.tool_call_name == "test_tool"
        assert result.tool_result_content == [{"type": "text", "text": "result"}]


class TestClient:
    def test_client_initialization(self, mock_mcp_server_config):
        """Test Client initialization with default parameters"""
        with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
            client = Client(mock_mcp_server_config)
            
            assert client.client_logger.name == "pocket_agent.client"
            assert client.mcp_logger.name == "pocket_agent.mcp"
            mock_mcp_client.assert_called_once()

    def test_client_initialization_with_custom_loggers(self, mock_mcp_server_config):
        """Test Client initialization with custom loggers"""
        custom_client_logger = logging.getLogger("custom.client")
        custom_mcp_logger = logging.getLogger("custom.mcp")
        custom_log_handler = Mock()
        
        with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
            client = Client(
                mock_mcp_server_config,
                client_logger=custom_client_logger,
                mcp_logger=custom_mcp_logger,
                mcp_log_handler=custom_log_handler
            )
            
            assert client.client_logger == custom_client_logger
            assert client.mcp_logger == custom_mcp_logger
            assert client.mcp_log_handler == custom_log_handler

    def test_default_mcp_log_handler(self, mock_mcp_server_config):
        """Test the default MCP log handler"""
        with patch('pocket_agent.client.MCPClient'):
            client = Client(mock_mcp_server_config)
            
            # Create a mock log message
            log_message = Mock()
            log_message.level = "INFO"
            log_message.data = {
                "msg": "Test message",
                "extra": {"key": "value"}
            }
            
            with patch.object(client.mcp_logger, 'log') as mock_log:
                client._default_mcp_log_handler(log_message)
                
                mock_log.assert_called_once_with(
                    logging.INFO, 
                    "[MCP] Test message", 
                    extra={"key": "value", "source": "mcp_server"}
                )

    @pytest.mark.asyncio
    async def test_get_mcp_tools(self, mock_mcp_server_config):
        """Test getting MCP tools"""
        mock_tools = [Mock(name="tool1"), Mock(name="tool2")]
        
        with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
            mock_client_instance = Mock()
            mock_client_instance.get_tools = AsyncMock(return_value=mock_tools)
            mock_mcp_client.return_value = mock_client_instance
            
            client = Client(mock_mcp_server_config)
            
            # Mock the async context manager
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            client.client = mock_client_instance
            
            result = await client._get_mcp_tools()
            
            assert result == mock_tools
            mock_client_instance.get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_openai_tools(self, mock_mcp_server_config):
        """Test converting MCP tools to OpenAI format"""
        mock_mcp_tools = [Mock(name="tool1")]
        mock_openai_tool = {"type": "function", "function": {"name": "tool1"}}
        
        with patch('pocket_agent.client.MCPClient'):
            with patch('pocket_agent.client.transform_mcp_tool_to_openai_tool', return_value=mock_openai_tool):
                client = Client(mock_mcp_server_config)
                
                with patch.object(client, '_get_mcp_tools', return_value=mock_mcp_tools):
                    result = await client._get_openai_tools()
                    
                    assert result == [mock_openai_tool]

    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_mcp_server_config):
        """Test successful tool call"""
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.name = "test_tool"
        mock_tool_call.model_dump.return_value = {"id": "call_123", "function": {"name": "test_tool"}}
        
        mock_mcp_result = Mock()
        mock_mcp_result.structuredContent = "Tool result"
        
        with patch('pocket_agent.client.MCPClient'):
            with patch('pocket_agent.client.transform_openai_tool_call_request_to_mcp_tool_call_request') as mock_transform:
                mock_mcp_request = Mock()
                mock_mcp_request.arguments = {"arg": "value"}
                mock_transform.return_value = mock_mcp_request
                
                client = Client(mock_mcp_server_config)
                client.client.call_tool = AsyncMock(return_value=mock_mcp_result)
                
                result = await client.call_tool(mock_tool_call)
                
                assert isinstance(result, ToolResult)
                assert result.tool_call_id == "call_123"
                assert result.tool_call_name == "test_tool"
                assert result.tool_result_content == [{"type": "text", "text": "Tool result"}]

    @pytest.mark.asyncio
    async def test_call_tool_with_tool_error(self, mock_mcp_server_config):
        """Test tool call with ToolError"""
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123" 
        mock_tool_call.name = "test_tool"
        mock_tool_call.model_dump.return_value = {"id": "call_123", "function": {"name": "test_tool"}}
        
        with patch('pocket_agent.client.MCPClient'):
            with patch('pocket_agent.client.transform_openai_tool_call_request_to_mcp_tool_call_request'):
                client = Client(mock_mcp_server_config)
                client.client.call_tool = AsyncMock(side_effect=ToolError("unexpected_keyword_argument test"))
                
                with patch.object(client, '_get_tool_format', return_value='{"arg": "string"}'):
                    result = await client.call_tool(mock_tool_call)
                    
                    assert isinstance(result, ToolResult)
                    assert "unexpected keyword argument" in result.tool_result_content[0]["text"]

    @pytest.mark.asyncio  
    async def test_call_tools_parallel(self, mock_mcp_server_config):
        """Test calling multiple tools in parallel"""
        mock_tool_calls = [Mock(), Mock()]
        mock_results = [Mock(), Mock()]
        
        with patch('pocket_agent.client.MCPClient'):
            client = Client(mock_mcp_server_config)
            
            # Use AsyncMock for the async method
            with patch.object(client, 'call_tool', new=AsyncMock(side_effect=mock_results)) as mock_call_tool:
                # Mock async context manager
                client.client.__aenter__ = AsyncMock(return_value=client.client)
                client.client.__aexit__ = AsyncMock(return_value=None)
                
                result = await client.call_tools(mock_tool_calls)
                
                assert result == mock_results
                # Verify call_tool was called for each tool_call
                assert mock_call_tool.call_count == len(mock_tool_calls)

    def test_parse_tool_result_with_structured_content(self, mock_mcp_server_config):
        """Test parsing tool result with structured content"""
        mock_result = Mock()
        mock_result.structuredContent = "Structured result"
        
        with patch('pocket_agent.client.MCPClient'):
            client = Client(mock_mcp_server_config)
            
            parsed = client._parse_tool_result(mock_result)
            
            assert parsed == [{"type": "text", "text": "Structured result"}]

    def test_parse_tool_result_with_content_array(self, mock_mcp_server_config):
        """Test parsing tool result with content array"""
        mock_text_content = Mock()
        mock_text_content.type = "text"
        mock_text_content.text = "Text content"
        
        mock_image_content = Mock()
        mock_image_content.type = "image"
        mock_image_content.imageBase64 = "base64data"
        
        mock_result = Mock()
        mock_result.structuredContent = None
        mock_result.content = [mock_text_content, mock_image_content]
        
        with patch('pocket_agent.client.MCPClient'):
            client = Client(mock_mcp_server_config)
            
            parsed = client._parse_tool_result(mock_result)
            
            assert len(parsed) == 2
            assert parsed[0] == {"type": "text", "text": "Text content"}
            assert parsed[1] == {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,base64data"}
            }

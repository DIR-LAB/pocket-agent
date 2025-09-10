import pytest
import logging
import uuid
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastmcp.client.logging import LogMessage
from fastmcp.exceptions import ToolError

from pocket_agent.agent import (
    MCPAgent, AgentConfig, GenerateMessageResult, 
    AgentEvent
)
from pocket_agent.client import Client, ToolResult



# Helper functions for common patterns
def create_mock_agent(mock_router, mock_mcp_server_config, sample_agent_config, mock_client=None):
    """Helper function to create a mocked agent with common setup"""
    with patch.object(MCPAgent, '_init_client') as mock_init_client:
        if mock_client is None:
            mock_client = Mock()
            mock_client._get_openai_tools = AsyncMock(return_value=[])
            mock_client.call_tools = AsyncMock(return_value=[])
        mock_init_client.return_value = mock_client
        
        return MCPAgent(
            Router=mock_router,
            mcp_server_config=mock_mcp_server_config,
            agent_config=sample_agent_config
        ), mock_client


def create_mock_client_with_patches(mock_mcp_server_config):
    """Helper function to create a mocked client with common patches"""
    with patch('pocket_agent.client.MCPClient') as mock_mcp_client:
        return Client(mock_mcp_server_config), mock_mcp_client


class TestAgentConfig:
    def test_agent_config_creation(self):
        """Test basic AgentConfig creation"""
        config = AgentConfig(
            llm_model="gpt-4",
            system_prompt="Test prompt"
        )
        assert config.llm_model == "gpt-4"
        assert config.system_prompt == "Test prompt"
        assert config.agent_id is None
        assert config.context_id is None
        assert config.messages == None
        assert config.allow_images is False

    def test_agent_config_with_all_fields(self):
        """Test AgentConfig with all fields populated"""
        messages = [{"role": "user", "content": "Hello"}]
        completion_kwargs = {"temperature": 0.7}
        
        config = AgentConfig(
            llm_model="gpt-3.5-turbo",
            agent_id="test-agent",
            context_id="test-context",
            system_prompt="You are helpful",
            messages=messages,
            allow_images=True,
            completion_kwargs=completion_kwargs
        )
        
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.agent_id == "test-agent"
        assert config.context_id == "test-context"
        assert config.system_prompt == "You are helpful"
        assert config.messages == messages
        assert config.allow_images is True
        assert config.get_completion_kwargs() == completion_kwargs

    def test_get_completion_kwargs_with_none(self):
        """Test get_completion_kwargs when completion_kwargs is None"""
        config = AgentConfig(llm_model="gpt-4")
        config.completion_kwargs = None
        assert config.get_completion_kwargs() == {}


class TestGenerateMessageResult:
    def test_generate_message_result_creation(self):
        """Test GenerateMessageResult dataclass"""
        result = GenerateMessageResult(
            message_content="Test content",
            image_base64s=["base64image"],
            reasoning_content="Test reasoning",
            thinking_blocks=[{"thought": "test"}],
            tool_calls=[{"function": {"name": "test_tool"}}]
        )
        
        assert result.message_content == "Test content"
        assert result.image_base64s == ["base64image"]
        assert result.reasoning_content == "Test reasoning"
        assert result.thinking_blocks == [{"thought": "test"}]
        assert result.tool_calls == [{"function": {"name": "test_tool"}}]


class TestAgentEvent:
    def test_agent_event_creation(self):
        """Test AgentEvent dataclass"""
        event = AgentEvent(
            event_type="new_message",
            data={"message": "test"}
        )
        
        assert event.event_type == "new_message"
        assert event.data == {"message": "test"}


class TestMCPAgent:
    def test_agent_initialization(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test MCPAgent initialization"""
        agent, _ = create_mock_agent(mock_router, mock_mcp_server_config, sample_agent_config)
        
        assert agent.Router == mock_router
        assert agent.agent_config == sample_agent_config
        assert agent.system_prompt == sample_agent_config.system_prompt
        assert agent.messages == sample_agent_config.messages
        assert agent.context_id == sample_agent_config.context_id
        assert agent.agent_id == sample_agent_config.agent_id

    def test_agent_initialization_generates_ids_when_none(self, mock_router, mock_mcp_server_config):
        """Test that agent generates IDs when not provided in config"""
        config = AgentConfig(llm_model="gpt-4")  # No IDs provided
        agent, _ = create_mock_agent(mock_router, mock_mcp_server_config, config)
        
        # Should have generated UUIDs
        assert agent.context_id is not None
        assert agent.agent_id is not None
        # Verify they're valid UUIDs
        uuid.UUID(agent.context_id)
        uuid.UUID(agent.agent_id)

    def test_format_messages(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test message formatting includes system prompt"""
        agent, _ = create_mock_agent(mock_router, mock_mcp_server_config, sample_agent_config)
        
        agent.messages = [{"role": "user", "content": "Hello"}]
        formatted = agent._format_messages()
        
        assert len(formatted) == 2
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == sample_agent_config.system_prompt
        assert formatted[1]["role"] == "user"
        assert formatted[1]["content"] == "Hello"

    def test_add_message(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test adding a message to the agent"""
        on_event_mock = Mock()
        
        with patch.object(MCPAgent, '_init_client'):
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=sample_agent_config,
                on_event=on_event_mock
            )
            
            message = {"role": "user", "content": "Test message"}
            agent.add_message(message)
            
            assert message in agent.messages
            on_event_mock.assert_called_once()

    def test_add_user_message_text_only(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test adding a user message with text only"""
        agent, _ = create_mock_agent(mock_router, mock_mcp_server_config, sample_agent_config)
        
        agent.add_user_message("Hello, agent!")
        
        assert len(agent.messages) == 1
        message = agent.messages[0]
        assert message["role"] == "user"
        assert message["content"][0]["type"] == "text"
        assert message["content"][0]["text"] == "Hello, agent!"

    def test_add_user_message_with_images_not_allowed(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test adding user message with images when images not allowed"""
        agent, _ = create_mock_agent(mock_router, mock_mcp_server_config, sample_agent_config)
        
        agent.add_user_message("Hello!", image_base64s=["base64data"])
        
        # Should only have text content, no images
        message = agent.messages[0]
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_generate_basic(self, mock_router, mock_mcp_server_config, sample_agent_config, sample_llm_response):
        """Test basic message generation"""
        agent, mock_client = create_mock_agent(mock_router, mock_mcp_server_config, sample_agent_config)
        mock_router.acompletion = AsyncMock(return_value=sample_llm_response)
        
        result = await agent.generate()
        
        assert isinstance(result, GenerateMessageResult)
        assert result.message_content == "Test response content"
        mock_router.acompletion.assert_called_once()

    def test_reset_messages(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test resetting messages"""
        agent, _ = create_mock_agent(mock_router, mock_mcp_server_config, sample_agent_config)
        
        agent.messages = [{"role": "user", "content": "test"}]
        agent.reset_messages()
        
        assert agent.messages == []

    def test_model_property(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test model property returns correct model name"""
        agent, _ = create_mock_agent(mock_router, mock_mcp_server_config, sample_agent_config)
        assert agent.model == sample_agent_config.llm_model

    def test_allow_images_property(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test allow_images property returns correct value"""
        agent, _ = create_mock_agent(mock_router, mock_mcp_server_config, sample_agent_config)
        assert agent.allow_images == sample_agent_config.allow_images


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
        client, mock_mcp_client = create_mock_client_with_patches(mock_mcp_server_config)
        
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
        client, _ = create_mock_client_with_patches(mock_mcp_server_config)
        
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
        
        client, _ = create_mock_client_with_patches(mock_mcp_server_config)
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
        
        client, _ = create_mock_client_with_patches(mock_mcp_server_config)
        parsed = client._parse_tool_result(mock_result)
        
        assert len(parsed) == 2
        assert parsed[0] == {"type": "text", "text": "Text content"}
        assert parsed[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,base64data"}
        }


class TestIntegration:
    @pytest.mark.asyncio
    async def test_agent_with_tool_calls(self, mock_router, mock_mcp_server_config):
        """Integration test: agent generates response with tool calls"""
        # Setup agent config
        config = AgentConfig(
            llm_model="gpt-4",
            system_prompt="You are a helpful assistant."
        )
        
        # Mock LLM response with tool calls
        mock_tool_call = {
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": '{"param": "value"}'},
            "type": "function"
        }
        
        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.model_dump.return_value = {
            "role": "assistant", 
            "tool_calls": [mock_tool_call]
        }
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_llm_response = Mock()
        mock_llm_response.choices = [mock_choice]
        
        # Mock tool execution result
        mock_tool_result = Mock()
        mock_tool_result.tool_call_id = "call_123"
        mock_tool_result.tool_call_name = "test_tool"
        mock_tool_result.tool_result_content = [{"type": "text", "text": "Tool executed successfully"}]
        
        mock_client = Mock()
        mock_client._get_openai_tools = AsyncMock(return_value=[])
        mock_client.call_tools = AsyncMock(return_value=[mock_tool_result])
        
        agent, _ = create_mock_agent(mock_router, mock_mcp_server_config, config, mock_client)
        mock_router.acompletion = AsyncMock(return_value=mock_llm_response)
        
        # Add user message
        agent.add_user_message("Please use the test tool.")
        
        # Execute one step (should generate response with tool call and execute it)
        await agent.step()
        
        # Verify the flow
        mock_router.acompletion.assert_called_once()
        mock_client.call_tools.assert_called_once()
        
        # Should have 3 messages: user, assistant with tool calls, and tool result
        assert len(agent.messages) == 3
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[1]["role"] == "assistant"
        assert agent.messages[2]["role"] == "tool"

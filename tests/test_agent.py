import pytest
import uuid
from unittest.mock import Mock, AsyncMock, patch
from pocket_agent.agent import (
    MCPAgent, AgentConfig, GenerateMessageResult, 
    AgentEvent
)


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
        with patch.object(MCPAgent, '_init_client') as mock_init_client:
            mock_client = Mock()
            mock_init_client.return_value = mock_client
            
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=sample_agent_config
            )
            
            assert agent.Router == mock_router
            assert agent.agent_config == sample_agent_config
            assert agent.system_prompt == sample_agent_config.system_prompt
            assert agent.messages == sample_agent_config.messages
            assert agent.context_id == sample_agent_config.context_id
            assert agent.agent_id == sample_agent_config.agent_id

    def test_agent_initialization_generates_ids_when_none(self, mock_router, mock_mcp_server_config):
        """Test that agent generates IDs when not provided in config"""
        config = AgentConfig(llm_model="gpt-4")  # No IDs provided
        
        with patch.object(MCPAgent, '_init_client'):
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=config
            )
            
            # Should have generated UUIDs
            assert agent.context_id is not None
            assert agent.agent_id is not None
            # Verify they're valid UUIDs
            uuid.UUID(agent.context_id)
            uuid.UUID(agent.agent_id)

    def test_format_messages(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test message formatting includes system prompt"""
        with patch.object(MCPAgent, '_init_client'):
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=sample_agent_config
            )
            
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
        with patch.object(MCPAgent, '_init_client'):
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=sample_agent_config
            )
            
            agent.add_user_message("Hello, agent!")
            
            assert len(agent.messages) == 1
            message = agent.messages[0]
            assert message["role"] == "user"
            assert message["content"][0]["type"] == "text"
            assert message["content"][0]["text"] == "Hello, agent!"

    def test_add_user_message_with_images_not_allowed(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test adding user message with images when images not allowed"""
        with patch.object(MCPAgent, '_init_client'):
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=sample_agent_config  # allow_images=False
            )
            
            agent.add_user_message("Hello!", image_base64s=["base64data"])
            
            # Should only have text content, no images
            message = agent.messages[0]
            assert len(message["content"]) == 1
            assert message["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_generate_basic(self, mock_router, mock_mcp_server_config, sample_agent_config, sample_llm_response):
        """Test basic message generation"""
        with patch.object(MCPAgent, '_init_client') as mock_init_client:
            mock_client = Mock()
            mock_client._get_openai_tools = AsyncMock(return_value=[])
            mock_init_client.return_value = mock_client
            mock_router.acompletion = AsyncMock(return_value=sample_llm_response)
            
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=sample_agent_config
            )
            
            result = await agent.generate()
            
            assert isinstance(result, GenerateMessageResult)
            assert result.message_content == "Test response content"
            mock_router.acompletion.assert_called_once()

    def test_reset_messages(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test resetting messages"""
        with patch.object(MCPAgent, '_init_client'):
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=sample_agent_config
            )
            
            agent.messages = [{"role": "user", "content": "test"}]
            agent.reset_messages()
            
            assert agent.messages == []

    def test_model_property(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test model property returns correct model name"""
        with patch.object(MCPAgent, '_init_client'):
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=sample_agent_config
            )
            
            assert agent.model == sample_agent_config.llm_model

    def test_allow_images_property(self, mock_router, mock_mcp_server_config, sample_agent_config):
        """Test allow_images property returns correct value"""
        with patch.object(MCPAgent, '_init_client'):
            agent = MCPAgent(
                Router=mock_router,
                mcp_server_config=mock_mcp_server_config,
                agent_config=sample_agent_config
            )
            
            assert agent.allow_images == sample_agent_config.allow_images
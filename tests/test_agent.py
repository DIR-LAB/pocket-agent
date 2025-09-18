import pytest
import uuid
from unittest.mock import Mock, AsyncMock, patch
from pocket_agent.agent import (
    PocketAgent, AgentConfig, 
    AgentEvent, StepResult, AgentHooks
)
from litellm.types.utils import ModelResponse, Message, Choices, Usage
import os
import asyncio


class SimpleTestAgent(PocketAgent):
    """Simple test agent that implements the abstract run method"""
    
    async def run(self) -> dict:
        """Simple run implementation for testing"""
        return {"status": "test_completed"}


class TestAgentConfig:
    """Test the AgentConfig dataclass"""

    def test_agent_config_creation(self):
        """Test basic AgentConfig creation with required fields"""
        config = AgentConfig(
            llm_model="gpt-4",
            system_prompt="Test prompt"
        )
        assert config.llm_model == "gpt-4"
        assert config.system_prompt == "Test prompt"
        assert config.agent_id is None
        assert config.messages is None
        assert config.allow_images is False

    def test_agent_config_with_all_fields(self):
        """Test AgentConfig with all fields populated"""
        messages = [{"role": "user", "content": "Hello"}]
        completion_kwargs = {"temperature": 0.7}
        
        config = AgentConfig(
            llm_model="gpt-3.5-turbo",
            agent_id="test-agent",
            system_prompt="You are helpful",
            messages=messages,
            allow_images=True,
            completion_kwargs=completion_kwargs
        )
        
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.agent_id == "test-agent"
        assert config.system_prompt == "You are helpful"
        assert config.messages == messages
        assert config.allow_images is True
        assert config.get_completion_kwargs() == completion_kwargs

    def test_get_completion_kwargs_handles_none(self):
        """Test get_completion_kwargs when completion_kwargs is None"""
        config = AgentConfig(llm_model="gpt-4")
        config.completion_kwargs = None
        assert config.get_completion_kwargs() == {}


class TestAgentEvent:
    """Test the AgentEvent dataclass"""
    
    def test_agent_event_creation(self):
        """Test AgentEvent creation"""
        event = AgentEvent(
            event_type="new_message",
            data={"message": "test"}
        )
        
        assert event.event_type == "new_message"
        assert event.data == {"message": "test"}


class TestStepResult:
    """Test the StepResult dataclass"""
    
    def test_step_result_creation(self):
        """Test StepResult creation"""
        mock_message = Message(
            content="Test response",
            role="assistant",
            tool_calls=None,
            function_call=None
        )
        
        result = StepResult(llm_message=mock_message)
        assert result.llm_message == mock_message
        assert result.tool_execution_results is None


class TestPocketAgent:
    """Simplified tests for PocketAgent core functionality"""

    @pytest.fixture
    def simple_config(self):
        """Simple agent configuration for testing"""
        return AgentConfig(
            llm_model="gpt-4",
            agent_id="test-agent",
            system_prompt="You are a test assistant",
            allow_images=False
        )

    @pytest.fixture
    def simple_mcp_config(self):
        """Simple MCP configuration for testing"""
        return {
            "mcpServers": {
                "test": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["server.py"],
                    "cwd": os.path.dirname(__file__)
                }
            }
        }

    def test_agent_initialization(self, simple_config, simple_mcp_config):
        """Test basic agent initialization"""
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=simple_mcp_config
        )
        
        assert agent.agent_id == "test-agent"
        assert agent.model == "gpt-4"
        assert agent.system_prompt == "You are a test assistant"
        assert agent.allow_images is False
        assert len(agent.messages) == 0
        assert agent.mcp_client is not None

    def test_agent_generates_id_when_none(self, simple_mcp_config):
        """Test that agent generates UUID when no agent_id provided"""
        config = AgentConfig(llm_model="gpt-4")  # No agent_id
        
        agent = SimpleTestAgent(
            agent_config=config,
            mcp_config=simple_mcp_config
        )
        
        # Should have generated a UUID
        assert agent.agent_id is not None
        uuid.UUID(agent.agent_id)  # Should not raise exception

    def test_agent_with_router(self, simple_config, simple_mcp_config, mock_router):
        """Test agent initialization with custom router"""
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=simple_mcp_config,
            router=mock_router
        )
        
        assert agent.llm_completion_handler == mock_router
        assert agent.agent_id == "test-agent"

    @pytest.mark.asyncio
    async def test_add_user_message_text_only(self, simple_config, simple_mcp_config):
        """Test adding a simple text message"""
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=simple_mcp_config
        )
        
        await agent.add_user_message("Hello, agent!")
        
        assert len(agent.messages) == 1
        message = agent.messages[0]
        assert message["role"] == "user"
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "text"
        assert message["content"][0]["text"] == "Hello, agent!"

    @pytest.mark.asyncio
    async def test_add_user_message_ignores_images_when_disabled(self, simple_config, simple_mcp_config):
        """Test that images are ignored when allow_images=False"""
        agent = SimpleTestAgent(
            agent_config=simple_config,  # allow_images=False
            mcp_config=simple_mcp_config
        )
        
        await agent.add_user_message("Hello!", image_base64s=["base64data"])
        
        # Should only have text content, no images
        message = agent.messages[0]
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "text"
        assert message["content"][0]["text"] == "Hello!"

    @pytest.mark.asyncio
    async def test_add_user_message_includes_images_when_enabled(self, simple_mcp_config):
        """Test that images are included when allow_images=True"""
        config = AgentConfig(
            llm_model="gpt-4",
            system_prompt="Test",
            allow_images=True  # Enable images
        )
        
        agent = SimpleTestAgent(
            agent_config=config,
            mcp_config=simple_mcp_config
        )
        
        await agent.add_user_message("Hello!", image_base64s=["base64data"])
        
        message = agent.messages[0]
        assert len(message["content"]) == 2  # text + image
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "image_url"
        assert "base64data" in message["content"][1]["image_url"]["url"]

    def test_format_messages_includes_system_prompt(self, simple_config, simple_mcp_config):
        """Test that _format_messages includes system prompt"""
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=simple_mcp_config
        )
        
        agent.messages = [{"role": "user", "content": "Hello"}]
        formatted = agent._format_messages()
        
        assert len(formatted) == 2
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == "You are a test assistant"
        assert formatted[1]["role"] == "user"
        assert formatted[1]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_basic_step_without_tools(self, simple_config, simple_mcp_config, mock_router):
        """Test basic step execution without tool calls"""
        # Mock simple LLM response
        mock_response = ModelResponse(
            id="test-123",
            created=1,
            model="gpt-4",
            object="chat.completion",
            system_fingerprint="test",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Hello! How can I help?",
                        role="assistant",
                        tool_calls=None,
                        function_call=None,
                    ),
                )
            ],
            usage=Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30),
            service_tier=None,
        )
        
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=simple_mcp_config,
            router=mock_router
        )
        
        await agent.add_user_message("Hello")
        result = await agent.step()
        
        assert isinstance(result, StepResult)
        assert result.llm_message.content == "Hello! How can I help?"
        assert result.llm_message.tool_calls is None
        assert result.tool_execution_results is None
        assert len(agent.messages) == 2  # user + assistant

    def test_reset_messages(self, simple_config, simple_mcp_config):
        """Test message reset functionality"""
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=simple_mcp_config
        )
        
        agent.messages = [{"role": "user", "content": "test"}]
        agent.reset_messages()
        
        assert agent.messages == []

    def test_agent_properties(self, simple_config, simple_mcp_config):
        """Test agent property accessors"""
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=simple_mcp_config
        )
        
        assert agent.model == "gpt-4"
        assert agent.allow_images is False
        assert agent.agent_id == "test-agent"

    @pytest.mark.asyncio
    async def test_agent_hooks_integration(self, simple_config, simple_mcp_config, mock_router):
        """Test that agent hooks work properly"""
        
        class TestHooks(AgentHooks):
            def __init__(self):
                self.events = []
                
            async def pre_step(self, context):
                self.events.append("pre_step")
                
            async def post_step(self, context):
                self.events.append("post_step")
                
            async def on_event(self, event):
                self.events.append(f"event:{event.event_type}")
        
        hooks = TestHooks()
        
        # Mock simple response
        mock_response = ModelResponse(
            id="test", created=1, model="gpt-4", object="chat.completion",
            system_fingerprint="test",
            choices=[
                Choices(
                    finish_reason="stop", index=0,
                    message=Message(content="Response", role="assistant", tool_calls=None, function_call=None)
                )
            ],
            usage=Usage(completion_tokens=5, prompt_tokens=10, total_tokens=15),
            service_tier=None,
        )
        
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=simple_mcp_config,
            router=mock_router,
            hooks=hooks
        )
        
        await agent.add_user_message("Test")
        await agent.step()
        
        # Verify hooks were called
        assert "pre_step" in hooks.events
        assert "post_step" in hooks.events
        assert any("event:new_message" in event for event in hooks.events)

    def test_run_method_implementation(self, simple_config, simple_mcp_config):
        """Test that the abstract run method is properly implemented"""
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=simple_mcp_config
        )
        
        # This should not raise an error since SimpleTestAgent implements run()
        result = asyncio.run(agent.run())
        assert result == {"status": "test_completed"}


class TestPocketAgentMultiAgent:
    """Test multi-agent functionality"""

    @pytest.fixture
    def simple_config(self):
        """Simple agent configuration for testing"""
        return AgentConfig(
            llm_model="gpt-4",
            agent_id="test-agent",
            name="TestAgent",
            system_prompt="You are a test assistant",
            allow_images=False
        )

    @pytest.fixture  
    def sub_agent_config(self):
        """Configuration for a sub-agent"""
        return AgentConfig(
            llm_model="gpt-3.5-turbo",
            name="SubAgent",
            role_description="A sub-agent that helps with specific tasks",
            system_prompt="You are a helpful sub-agent",
            allow_images=False
        )

    @pytest.fixture
    def simple_mcp_config(self):
        """Simple MCP configuration for testing"""
        return {
            "mcpServers": {
                "test": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["server.py"],
                    "cwd": os.path.dirname(__file__)
                }
            }
        }

    def test_agent_works_with_sub_agents_only(self, simple_config, sub_agent_config, simple_mcp_config):
        """Test that agent works when no mcp_config is provided but sub_agents are"""
        # First create a sub-agent with its own mcp_config
        sub_agent = SimpleTestAgent(
            agent_config=sub_agent_config,
            mcp_config=simple_mcp_config
        )
        
        # Create main agent with only sub_agents, no mcp_config
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=None,  # Explicitly no MCP config
            sub_agents=[sub_agent]
        )
        
        assert agent.agent_id == "test-agent"
        assert agent.model == "gpt-4"
        assert agent.has_sub_agents is True
        assert agent.sub_agent_count == 1
        assert agent.mcp_client is not None
        assert agent.sub_agents[0].is_sub_agent is True

    def test_agent_fails_when_neither_config_provided(self, simple_config):
        """Test that agent fails when neither mcp_config nor sub_agents are provided"""
        with pytest.raises(ValueError) as exc_info:
            SimpleTestAgent(
                agent_config=simple_config,
                mcp_config=None,
                sub_agents=None
            )
        
        assert "MCP config is empty and no sub agents are provided" in str(exc_info.value)
        assert "At least one of the two must be provided" in str(exc_info.value)

    def test_agent_works_with_both_mcp_config_and_sub_agents(self, simple_config, sub_agent_config, simple_mcp_config):
        """Test that agent works when both mcp_config and sub_agents are provided"""
        # Create sub-agent with its own config
        sub_agent = SimpleTestAgent(
            agent_config=sub_agent_config,
            mcp_config=simple_mcp_config
        )
        
        # Create main agent with both mcp_config and sub_agents
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=simple_mcp_config,
            sub_agents=[sub_agent]
        )
        
        assert agent.agent_id == "test-agent"
        assert agent.model == "gpt-4"
        assert agent.has_sub_agents is True
        assert agent.sub_agent_count == 1
        assert agent.mcp_client is not None
        assert agent.sub_agents[0].is_sub_agent is True

    def test_agent_with_multiple_sub_agents(self, simple_config, simple_mcp_config):
        """Test agent with multiple sub-agents"""
        # Create multiple sub-agents
        sub_agent_1 = SimpleTestAgent(
            agent_config=AgentConfig(
                llm_model="gpt-3.5-turbo",
                name="SubAgent1",
                role_description="First sub-agent",
                system_prompt="You are sub-agent 1"
            ),
            mcp_config=simple_mcp_config
        )
        
        sub_agent_2 = SimpleTestAgent(
            agent_config=AgentConfig(
                llm_model="gpt-3.5-turbo", 
                name="SubAgent2",
                role_description="Second sub-agent",
                system_prompt="You are sub-agent 2"
            ),
            mcp_config=simple_mcp_config
        )
        
        # Create main agent with multiple sub-agents
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=None,
            sub_agents=[sub_agent_1, sub_agent_2]
        )
        
        assert agent.has_sub_agents is True
        assert agent.sub_agent_count == 2
        assert agent.sub_agents[0].is_sub_agent is True
        assert agent.sub_agents[1].is_sub_agent is True
        assert agent.sub_agents[0].name == "SubAgent1"
        assert agent.sub_agents[1].name == "SubAgent2"

    def test_sub_agent_properties(self, simple_config, sub_agent_config, simple_mcp_config):
        """Test sub-agent specific properties"""
        # Create sub-agent
        sub_agent = SimpleTestAgent(
            agent_config=sub_agent_config,
            mcp_config=simple_mcp_config
        )
        
        # Create main agent
        agent = SimpleTestAgent(
            agent_config=simple_config,
            mcp_config=None,
            sub_agents=[sub_agent]
        )
        
        # Test main agent properties
        assert agent.has_sub_agents is True
        assert agent.sub_agent_count == 1
        assert agent.is_sub_agent is False  # Main agent is not a sub-agent
        
        # Test sub-agent properties  
        assert sub_agent.has_sub_agents is False  # Sub-agent has no sub-agents
        assert sub_agent.sub_agent_count == 0
        assert sub_agent.is_sub_agent is True  # Sub-agent is marked as such
        assert sub_agent.role_description == "A sub-agent that helps with specific tasks"
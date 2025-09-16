"""
Simplified integration tests for the pocket agent framework.

These tests verify end-to-end functionality using real FastMCP servers
instead of complex mocking.
"""

import pytest
import asyncio
import os
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

from pocket_agent.agent import PocketAgent, AgentConfig, AgentHooks, StepResult
from pocket_agent.client import PocketAgentClient
from litellm.types.utils import ModelResponse, Message, Choices, Usage
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport


class SimpleIntegrationTestAgent(PocketAgent):
    """Simplified test agent for integration testing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_results = []
    
    async def run_conversation(self, messages: list[str], max_steps: int = 5) -> list[StepResult]:
        """Run a conversation and return all step results"""
        results = []
        
        for message in messages:
            await self.add_user_message(message)
            
            step_count = 0
            while step_count < max_steps:
                step_result = await self.step()
                results.append(step_result)
                
                # Stop if no tool calls in the last message
                if not step_result.llm_message.tool_calls:
                    break
                    
                step_count += 1
        
        self.step_results = results
        return results

    async def run(self) -> dict:
        """Simple run implementation for testing"""
        return {"status": "test_completed"}


class TestAgentIntegration:
    """Simplified integration tests using real FastMCP servers"""

    @pytest.fixture
    def simple_agent_config(self):
        """Simple agent configuration for testing"""
        return AgentConfig(
            llm_model="gpt-4",
            agent_id="test-agent",
            system_prompt="You are a helpful assistant that can use tools.",
            allow_images=False,
            completion_kwargs={"tool_choice": "auto"}
        )

    @pytest.fixture
    def real_mcp_config(self):
        """MCP configuration that points to real test server"""
        return {
            "mcpServers": {
                "test_server": {
                    "transport": "stdio",
                    "command": "python",
                    "args":["server.py"],
                    "cwd": os.path.dirname(__file__)
                }
            }
        }

    @pytest.fixture
    def mock_llm_response_no_tools(self):
        """Simple LLM response without tool calls"""
        return ModelResponse(
            id="test-123",
            created=1735081811,
            model="gpt-4",
            object="chat.completion",
            system_fingerprint="test",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Hello! I'm ready to help.",
                        role="assistant",
                        tool_calls=None,
                        function_call=None,
                    ),
                )
            ],
            usage=Usage(completion_tokens=10, prompt_tokens=20, total_tokens=30),
            service_tier=None,
        )

    @pytest.fixture  
    def mock_llm_response_with_greet_tool(self):
        """LLM response that wants to call the greet tool"""
        from litellm.types.utils import ChatCompletionMessageToolCall, Function
        
        return ModelResponse(
            id="test-456",
            created=1735081811,
            model="gpt-4", 
            object="chat.completion",
            system_fingerprint="test",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                function=Function(
                                    arguments='{"name":"Alice"}',
                                    name="greet",
                                ),
                                id="call_greet_123",
                                type="function",
                            )
                        ],
                        function_call=None,
                    ),
                )
            ],
            usage=Usage(completion_tokens=25, prompt_tokens=30, total_tokens=55),
            service_tier=None,
        )

    @pytest.fixture
    def mock_llm_final_response(self):
        """Final LLM response after tool calls"""
        return ModelResponse(
            id="test-final",
            created=1735081811,
            model="gpt-4",
            object="chat.completion",
            system_fingerprint="test",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="I've completed the requested tasks!",
                        role="assistant", 
                        tool_calls=None,
                        function_call=None,
                    ),
                )
            ],
            usage=Usage(completion_tokens=10, prompt_tokens=50, total_tokens=60),
            service_tier=None,
        )

    @pytest.mark.asyncio
    async def test_agent_initialization_with_real_server(self, real_mcp_config, simple_agent_config):
        """Test agent initialization with real MCP server"""
        agent = SimpleIntegrationTestAgent(
            agent_config=simple_agent_config,
            mcp_config=real_mcp_config
        )
        
        # Basic initialization checks
        assert agent.agent_id == "test-agent"
        assert agent.model == "gpt-4"
        assert len(agent.messages) == 0
        assert agent.allow_images is False
        assert agent.mcp_client is not None

    @pytest.mark.asyncio
    async def test_fastmcp_server_direct_integration(self, fastmcp_server):
        """Test direct integration with FastMCP server using in-memory transport"""
        client = Client(transport=FastMCPTransport(fastmcp_server))
        
        async with client:
            # Test listing tools
            tools = await client.list_tools()
            assert len(tools) == 3
            tool_names = {tool.name for tool in tools}
            assert tool_names == {"greet", "add", "sleep"}
            
            # Test calling tools
            result = await client.call_tool("greet", {"name": "World"})
            assert "Hello, World!" in str(result.content)
            
            result = await client.call_tool("add", {"a": 5, "b": 3})
            assert "8" in str(result.content)

    @pytest.mark.asyncio
    async def test_real_mcp_server_tool_listing(self, real_mcp_config):
        """Test that we can list tools from the real MCP server"""
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        # Get tools from the real server
        tools = await client.get_tools(format="mcp")
        
        assert len(tools) >= 3  # Should have at least greet, add, sleep
        tool_names = {tool.name for tool in tools}
        assert "greet" in tool_names
        assert "add" in tool_names
        assert "sleep" in tool_names

    @pytest.mark.asyncio
    async def test_real_mcp_server_tool_calling(self, real_mcp_config):
        """Test calling tools on the real MCP server"""
        client = PocketAgentClient(mcp_config=real_mcp_config)
        
        # Test calling the greet tool
        from mcp.types import CallToolRequestParams
        greet_call = CallToolRequestParams(name="greet", arguments={"name": "Test User"})
        greet_call.id = "test_call_1"
        
        result = await client.call_tool(greet_call)
        
        assert result.tool_call_id == "test_call_1"
        assert result.tool_call_name == "greet" 
        assert "Hello, Test User!" in result.tool_result_content[0]["text"]

    @pytest.mark.asyncio
    async def test_simple_conversation_flow(self, real_mcp_config, simple_agent_config, mock_router):
        """Test a simple conversation without tools"""
        mock_response = ModelResponse(
            id="test",
            created=1,
            model="gpt-4",
            object="chat.completion", 
            system_fingerprint="test",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="I understand! How can I help?",
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
        
        agent = SimpleIntegrationTestAgent(
            agent_config=simple_agent_config,
            mcp_config=real_mcp_config,
            router=mock_router
        )
        
        # Test adding a message and getting response
        await agent.add_user_message("Hello!")
        step_result = await agent.step()
        
        assert step_result.llm_message.content == "I understand! How can I help?"
        assert step_result.llm_message.tool_calls is None
        assert len(agent.messages) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_tool_calling_workflow(self, real_mcp_config, simple_agent_config, mock_router):
        """Test complete tool calling workflow with real server"""
        # Mock LLM to first request a tool call, then provide final response
        from litellm.types.utils import ChatCompletionMessageToolCall, Function
        mock_tool_response = ModelResponse(
            id="test-tool",
            created=1,
            model="gpt-4",
            object="chat.completion",
            system_fingerprint="test",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                function=Function(arguments='{"name":"Integration Test"}', name="greet"),
                                id="call_123",
                                type="function",
                            )
                        ],
                        function_call=None,
                    ),
                )
            ],
            usage=Usage(completion_tokens=25, prompt_tokens=30, total_tokens=55),
            service_tier=None,
        )
        
        mock_final_response = ModelResponse(
            id="test-final",
            created=1,
            model="gpt-4",
            object="chat.completion",
            system_fingerprint="test", 
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="I've greeted the user successfully!",
                        role="assistant",
                        tool_calls=None,
                        function_call=None,
                    ),
                )
            ],
            usage=Usage(completion_tokens=10, prompt_tokens=50, total_tokens=60),
            service_tier=None,
        )
        
        # Mock router to return tool call first, then final response
        mock_router.acompletion = AsyncMock(side_effect=[mock_tool_response, mock_final_response])
        
        agent = SimpleIntegrationTestAgent(
            agent_config=simple_agent_config,
            mcp_config=real_mcp_config,
            router=mock_router
        )
        
        # Run conversation
        results = await agent.run_conversation(["Please greet 'Integration Test'"])
        
        # Should have 2 steps: tool call + final response
        assert len(results) == 2
        
        # First step should have tool calls and results
        assert results[0].llm_message.tool_calls is not None
        assert len(results[0].tool_execution_results) == 1
        assert "Hello, Integration Test!" in results[0].tool_execution_results[0].tool_result_content[0]["text"]
        
        # Second step should be final response
        assert results[1].llm_message.tool_calls is None
        assert results[1].llm_message.content == "I've greeted the user successfully!"

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_concept(self, fastmcp_server):
        """Demonstrate parallel tool calling with in-memory server"""
        client = Client(transport=FastMCPTransport(fastmcp_server))
        
        async with client:
            # Execute multiple tools in parallel
            tasks = [
                client.call_tool("greet", {"name": "Alice"}),
                client.call_tool("add", {"a": 10, "b": 20}),
                client.call_tool("greet", {"name": "Bob"})
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert "Hello, Alice!" in str(results[0].content)
            assert "30" in str(results[1].content)
            assert "Hello, Bob!" in str(results[2].content)

    @pytest.mark.asyncio
    async def test_error_handling_with_real_server(self, real_mcp_config, simple_agent_config):
        """Test error handling with real MCP server"""
        agent = SimpleIntegrationTestAgent(
            agent_config=simple_agent_config,
            mcp_config=real_mcp_config
        )
        
        # Test calling tool with wrong parameters
        from mcp.types import CallToolRequestParams
        bad_call = CallToolRequestParams(name="greet", arguments={"greet": "test", "name": "test"})
        bad_call.id = "bad_call"
        
        # This should handle the error gracefully (depending on implementation)
        result = await agent.mcp_client.call_tool(bad_call)
        # The exact error handling behavior depends on the implementation
        assert result.tool_call_id == "bad_call"

    @pytest.mark.asyncio
    async def test_agent_hooks_simple(self, simple_agent_config, real_mcp_config, mock_router):
        """Test agent hooks with minimal setup"""
        
        class SimpleHooks(AgentHooks):
            def __init__(self):
                self.events = []
                self.step_count = 0
            
            async def pre_step(self, context):
                self.step_count += 1
                self.events.append("pre_step")
            
            async def post_step(self, context):
                self.events.append("post_step")
        
        hooks = SimpleHooks()
        
        # Mock simple response
        mock_response = ModelResponse(
            id="test", created=1, model="gpt-4", object="chat.completion",
            system_fingerprint="test",
            choices=[
                Choices(
                    finish_reason="stop", index=0,
                    message=Message(content="Hello!", role="assistant", tool_calls=None, function_call=None)
                )
            ],
            usage=Usage(completion_tokens=5, prompt_tokens=10, total_tokens=15),
            service_tier=None,
        )
        
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        
        agent = SimpleIntegrationTestAgent(
            agent_config=simple_agent_config,
            mcp_config=real_mcp_config,
            router=mock_router,
            hooks=hooks
        )
        
        await agent.add_user_message("Test message")
        await agent.step()
        
        # Verify hooks were called
        assert hooks.step_count == 1
        assert "pre_step" in hooks.events
        assert "post_step" in hooks.events

    @pytest.mark.asyncio
    async def test_message_management(self, simple_agent_config, real_mcp_config):
        """Test basic message management functionality"""
        agent = SimpleIntegrationTestAgent(
            agent_config=simple_agent_config,
            mcp_config=real_mcp_config
        )
        
        # Test adding messages
        await agent.add_user_message("First message")
        await agent.add_user_message("Second message")
        
        assert len(agent.messages) == 2
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[1]["role"] == "user"
        assert agent.messages[0]["content"][0]["text"] == "First message"
        
        # Test reset
        agent.reset_messages()
        assert len(agent.messages) == 0

    @pytest.mark.asyncio
    async def test_image_handling_disabled(self, simple_agent_config, real_mcp_config):
        """Test that images are ignored when disabled"""
        agent = SimpleIntegrationTestAgent(
            agent_config=simple_agent_config,  # allow_images=False by default
            mcp_config=real_mcp_config
        )
        
        # Add message with image - should be ignored
        await agent.add_user_message("Describe image", image_base64s=["fake_base64"])
        
        # Verify only text was added
        user_message = agent.messages[-1]
        assert len(user_message["content"]) == 1
        assert user_message["content"][0]["type"] == "text"
        assert user_message["content"][0]["text"] == "Describe image"

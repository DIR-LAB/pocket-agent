# Pocket Agent Test Suite Documentation

This document describes all the tests in the Pocket Agent test suite, organized by test file.

## Table of Contents

1. [conftest.py](#conftest.py) - Test Configuration and Fixtures
2. [test_agent.py](#test_agent.py) - Agent Module Tests  
3. [test_client.py](#test_client.py) - Client Module Tests
4. [test_integration.py](#test_integration.py) - Integration Tests
5. [test_mcp_server.py](#test_mcp_server.py) - MCP Server Integration Tests

---

## conftest.py

Contains shared test fixtures and configuration used across all test files.

### Fixtures

- **`sample_agent_config()`**: Creates a basic `AgentConfig` for testing with gpt-5 model, test IDs, system prompt, empty messages, and completion kwargs.

- **`mock_router()`**: Creates a mock LiteLLM Router with mocked `acompletion` method for testing LLM interactions without real API calls.

- **`mock_mcp_server_config()`**: Provides mock MCP server configuration with dummy server command and args.

- **`fastmcp_server()`**: Creates a complete FastMCP server with:
  - **Tools**: `greet(name)` - greets someone, `add(a,b)` - adds numbers, `sleep(seconds)` - async sleep 
  - **Resources**: `data://users` - returns user list, `data://user/{user_id}` - returns user data
  - **Prompts**: `welcome(name)` - generates welcome message

- **`mock_logger()`**: Mock logger for testing logging functionality.

- **`mock_client()`**: Mock MCP client with mocked `_get_openai_tools()` and `call_tools()` methods.

- **`sample_llm_response()`**: Creates a mock LLM response object with test content and proper structure for testing agent message generation.

---

## test_agent.py

Tests the core agent functionality and data classes.

### TestAgentConfig

Tests for the `AgentConfig` data class:

- **`test_agent_config_creation()`**: Verifies basic `AgentConfig` can be created with required fields (llm_model, system_prompt) and that optional fields default correctly (agent_id=None, context_id=None, messages=None, allow_images=False).

- **`test_agent_config_with_all_fields()`**: Tests `AgentConfig` creation with all fields populated including messages list, completion_kwargs, and image support enabled.

- **`test_get_completion_kwargs_with_none()`**: Verifies that `get_completion_kwargs()` returns empty dict when `completion_kwargs` is None.

### TestGenerateMessageResult

- **`test_generate_message_result_creation()`**: Tests the `GenerateMessageResult` dataclass can be created with all fields (message_content, image_base64s, reasoning_content, thinking_blocks, tool_calls) and that values are stored correctly.

### TestAgentEvent

- **`test_agent_event_creation()`**: Tests the `AgentEvent` dataclass creation with event_type and data fields.

### TestMCPAgent

Tests for the main `MCPAgent` class:

- **`test_agent_initialization()`**: Verifies `MCPAgent` initializes correctly with Router, MCP server config, and agent config. Checks that all config values are properly assigned to agent properties.

- **`test_agent_initialization_generates_ids_when_none()`**: Tests that when agent_id and context_id are not provided in config, the agent generates valid UUIDs for these fields.

- **`test_format_messages()`**: Verifies that `_format_messages()` properly formats messages by prepending the system prompt as the first message with role "system".

- **`test_add_message()`**: Tests that `add_message()` adds a message to the agent's message list and triggers the event callback if provided.

- **`test_add_user_message_text_only()`**: Tests `add_user_message()` with text-only input creates properly structured user message with content array containing text object.

- **`test_add_user_message_with_images_not_allowed()`**: Verifies that when `allow_images=False`, images passed to `add_user_message()` are ignored and only text content is added.

- **`test_generate_basic()`**: Tests basic message generation flow - mocks client tools and LLM response, calls `generate()`, and verifies it returns a `GenerateMessageResult` with correct content.

- **`test_reset_messages()`**: Tests that `reset_messages()` clears the agent's message list.

- **`test_model_property()`**: Verifies the `model` property returns the correct LLM model name from the config.

- **`test_allow_images_property()`**: Verifies the `allow_images` property returns the correct value from the config.

---

## test_client.py

Tests the MCP client functionality for tool integration.

### TestToolResult

- **`test_tool_result_creation()`**: Tests the `ToolResult` dataclass creation with tool_call_id, tool_call_name, and tool_result_content fields.

### TestClient

Tests for the `Client` class that handles MCP server communication:

- **`test_client_initialization()`**: Tests `Client` initialization with default parameters, verifies correct logger names are set and MCP client is created.

- **`test_client_initialization_with_custom_loggers()`**: Tests `Client` initialization with custom client logger, MCP logger, and log handler.

- **`test_default_mcp_log_handler()`**: Tests the default MCP log message handler formats log messages correctly with "[MCP]" prefix and adds "mcp_server" source to extra data.

- **`test_get_mcp_tools()`**: Tests `_get_mcp_tools()` method retrieves tools from MCP server using async context manager.

- **`test_get_openai_tools()`**: Tests `_get_openai_tools()` method converts MCP tools to OpenAI tool format using transformation function.

- **`test_call_tool_success()`**: Tests successful tool execution - mocks tool call, MCP result, and transformation, verifies `call_tool()` returns proper `ToolResult`.

- **`test_call_tool_with_tool_error()`**: Tests tool call error handling when `ToolError` is raised, verifies error message is returned in tool result instead of raising exception.

- **`test_call_tools_parallel()`**: Tests `call_tools()` method calls multiple tools in parallel using async context manager and returns all results.

- **`test_parse_tool_result_with_structured_content()`**: Tests `_parse_tool_result()` with structured content returns properly formatted text result.

- **`test_parse_tool_result_with_content_array()`**: Tests `_parse_tool_result()` with mixed content types (text and image) returns properly formatted results with image URLs.

---

## test_integration.py

Contains integration tests and duplicates some unit tests with helper functions.

### Helper Functions

- **`create_mock_agent()`**: Helper that creates a mocked `MCPAgent` with common setup including mocked client and tools.

- **`create_mock_client_with_patches()`**: Helper that creates a mocked `Client` with common patches applied.

### Test Classes

The file duplicates most test classes from `test_agent.py` and `test_client.py` but uses the helper functions for cleaner setup.

### TestIntegration

- **`test_agent_with_tool_calls()`**: Comprehensive integration test that:
  - Creates agent with sample config
  - Mocks LLM response with tool calls
  - Mocks tool execution results  
  - Tests complete workflow: user message → agent.step() → tool execution → message history
  - Verifies proper message flow with 3 messages: user, assistant with tool calls, and tool result

---

## test_mcp_server.py

Tests integration with FastMCP server functionality.

### TestFastMCPServerIntegration

Comprehensive tests for MCP server integration:

- **`test_client_get_tools_from_mcp_server()`**: Tests that `Client` can retrieve tools from MCP server. Mocks 3 tools (greet, add, sleep) and verifies all are returned with correct names.

- **`test_client_get_openai_tools()`**: Tests MCP to OpenAI tool format conversion. Mocks MCP tool and transformation, verifies OpenAI format structure.

- **`test_client_call_greet_tool()`**: Tests calling the "greet" tool through client:
  - Mocks tool call with name parameter
  - Mocks MCP result with greeting message
  - Verifies `ToolResult` is returned with correct content

- **`test_client_call_add_tool()`**: Tests calling the "add" tool through client:
  - Mocks tool call with a=3, b=4 parameters
  - Mocks MCP result returning "7"
  - Verifies correct tool execution and result formatting

- **`test_client_call_multiple_tools_parallel()`**: Tests parallel execution of multiple tools:
  - Creates mock calls for both "greet" and "add" tools
  - Uses side_effect to return different results based on tool name
  - Verifies both tools execute correctly in parallel

- **`test_agent_with_mcp_server_tools()`**: Complete end-to-end workflow test:
  - Sets up agent with MCP server tools (greet, add)
  - Mocks LLM response requesting both tools
  - Tests agent.step() handles tool calls and execution
  - Verifies complete conversation flow with user message, assistant message, and tool result messages

- **`test_tool_error_handling()`**: Tests error handling for tool failures:
  - Mocks tool that raises `ToolError` with "unexpected_keyword_argument" 
  - Verifies error is caught and returned as `ToolResult` with error message
  - Tests that tool format information is included in error response

- **`test_parse_tool_result_with_content_array()`**: Tests parsing of complex tool results:
  - Creates mock result with text and image content
  - Verifies both content types are properly parsed
  - Tests image content is converted to proper image_url format with base64 data URL

---

## Test Coverage Summary

The test suite provides comprehensive coverage of:

1. **Data Classes**: All dataclasses (`AgentConfig`, `GenerateMessageResult`, `AgentEvent`, `ToolResult`) are tested for proper creation and field access.

2. **Agent Functionality**: Core agent operations like initialization, message handling, generation, and configuration are thoroughly tested.

3. **Client Operations**: MCP client functionality including tool retrieval, execution, error handling, and result parsing.

4. **Integration Flows**: End-to-end workflows combining agent and client functionality with real tool execution scenarios.

5. **Error Handling**: Various error conditions are tested to ensure robust error handling and graceful degradation.

6. **Async Operations**: Async functionality is properly tested using pytest-asyncio markers.

7. **Mocking Strategy**: Extensive use of mocks allows testing without external dependencies while maintaining realistic test scenarios.

# Test Suite Documentation

This directory contains comprehensive tests for the pocket agent framework.

## Overview

The test suite verifies functionality across multiple layers of the pocket agent framework, including:

- Agent initialization and configuration
- Client-server interactions with MCP servers
- Tool execution and result handling
- Message management and conversation flows
- Error handling and recovery scenarios

## Test Files

### `test_agent.py`
Core agent functionality tests containing:
- **TestAgentConfig**: Configuration validation and creation
- **TestAgentEvent**: Event handling and creation
- **TestStepResult**: Step execution result handling
- **TestPocketAgent**: Core agent behavior including initialization, message handling, and step execution

### `test_client.py`
Client functionality tests containing:
- **TestToolResult**: Tool result data structure validation
- **TestPocketAgentClient**: Client operations including tool discovery, execution, and format handling

### `test_mcp_server.py`
MCP server integration tests containing:
- **TestFastMCPServerIntegration**: Direct integration with FastMCP servers using in-memory transport

### `test_integration.py`
End-to-end integration tests containing:
- **TestAgentIntegration**: Complete agent workflow testing including conversation flows, tool calls, and hook system integration

### `conftest.py`
Shared fixtures and configuration for all tests, including:
- `fastmcp_server`: In-memory FastMCP server with sample tools (greet, add, sleep)
- `mock_router`: Mock LiteLLM Router for testing without external API calls
- Router configuration constants

### `server.py`
Standalone test server implementation used by some client tests for stdio transport testing.

## Running Tests

### Using the Test Runner Script (Recommended)

The easiest way to run tests is using the provided script:

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run with coverage reporting
python run_tests.py --coverage

# Run quick subset for development
python run_tests.py --quick

# Run tests in parallel (experimental)
python run_tests.py --parallel
```

### Using uv directly

```bash
# Install test dependencies
uv sync --group test

# Run all tests
uv run pytest tests/ -v --asyncio-mode=auto

# Run specific test file
uv run pytest tests/test_agent.py -v --asyncio-mode=auto

# Run with coverage
uv run pytest tests/ --cov=pocket_agent --cov-report=html --cov-report=term
```

### Using pytest directly

```bash
# After installing dependencies with uv sync --group test
pytest tests/ -v --asyncio-mode=auto
```

## Test Structure

### Test Classes Overview

1. **TestAgentConfig**: Agent configuration validation
   - Configuration creation and validation
   - Default value handling
   - Field validation

2. **TestPocketAgent**: Core agent functionality
   - Agent initialization with various configurations
   - Message handling (text and images)
   - Step execution without tool calls
   - Hook system integration
   - Property access and configuration

3. **TestPocketAgentClient**: Client-server interactions
   - Tool discovery and listing
   - Tool execution (sync and async)
   - Format transformations (MCP ↔ OpenAI)
   - Error handling

4. **TestAgentIntegration**: End-to-end workflows
   - Complete conversation flows
   - Tool call execution
   - Multi-step agent interactions
   - Real server integration testing

### Test Patterns

The tests use several key patterns:

- **In-memory testing**: FastMCP servers with in-memory transport for fast, reliable testing
- **Mock-based LLM responses**: Controlled LLM responses using mocks instead of external API calls
- **Async test support**: All async functionality properly tested with `@pytest.mark.asyncio`
- **Fixture reuse**: Shared fixtures in `conftest.py` for common test setup
- **Real server integration**: Some tests use actual stdio transport servers for realistic scenarios

## Key Features Tested

### Agent Lifecycle
- Initialization with various configurations
- Message history management
- Step execution and result processing
- Agent ID generation and handling

### Tool Integration
- Tool discovery from MCP servers
- Single and async tool execution
- Tool result processing and formatting
- Error handling for tool failures

### Message Handling
- Text message processing
- Image handling (when enabled)
- Message formatting and validation
- Conversation flow management

### Hook System
- Pre/post step hooks
- Event handling and propagation
- Custom hook implementations

### Error Handling
- Configuration validation errors
- Tool execution failures
- Invalid input handling
- Mock error scenarios

## Development Guidelines

### Adding New Tests

1. **Follow existing patterns**: Use similar structure and naming conventions
2. **Use appropriate test class**: Add tests to the most relevant existing test class
3. **Mock external dependencies**: Use mocks for LLM calls and external services
4. **Test both success and failure paths**: Include error scenarios
5. **Use descriptive test names**: Clearly indicate what functionality is being tested

### Fixture Usage

The `conftest.py` provides shared fixtures:
- Use `fastmcp_server` for testing tool interactions
- Use `mock_router` for testing without external LLM API calls
- Create local fixtures for test-specific setup

### Async Testing

All async tests should:
- Use `@pytest.mark.asyncio` decorator
- Properly handle async context managers
- Use `AsyncMock` for async function mocks
- Include proper cleanup of async resources

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `uv sync --group test` has been run
2. **Async test failures**: Make sure `--asyncio-mode=auto` is used with pytest
3. **Mock setup errors**: Verify mock configuration matches actual interfaces
4. **Server connection issues**: Check that test servers are properly configured

### Debug Mode

For debugging test failures:

```bash
# Run with maximum verbosity
python run_tests.py --verbose

# Run specific failing test
uv run pytest tests/test_agent.py::TestPocketAgent::test_specific_test -v -s

# Run with pdb debugging
uv run pytest tests/test_agent.py --pdb
```

## Coverage

The test suite aims for comprehensive coverage of:
- Core agent functionality
- Client-server interactions
- Configuration handling
- Error scenarios
- Tool integration workflows

Generate coverage reports with:
```bash
python run_tests.py --coverage
```

The HTML coverage report will be available in `htmlcov/index.html`.
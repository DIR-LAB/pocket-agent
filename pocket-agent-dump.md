# Complete Documentation & Source Code of Pocket Agent

**Pocket Agent Version:** v0.2.23  
**Generated on:** 2025-11-24 20:06:53  
**Total Files:** 14

## 📋 Table of Contents

### 📖 Documentation

- 📖 [Index](#docs-00-index-md) *(47 lines)*
- 📖 [Getting Started](#docs-01-getting-started-md) *(64 lines)*
- 📖 [Core Concepts](#docs-02-core-concepts-md) *(90 lines)*
- 📖 [Hooks And Events](#docs-03-hooks-and-events-md) *(95 lines)*
- 📖 [Multi Model Support](#docs-04-multi-model-support-md) *(46 lines)*
- 📖 [Client](#docs-05-client-md) *(115 lines)*
- 📖 [Multi Agent](#docs-06-multi-agent-md) *(92 lines)*
- 📖 [Agent As A Server](#docs-07-agent-as-a-server-md) *(19 lines)*
- 📖 [Testing](#docs-08-testing-md) *(22 lines)*

### 🔧 Source Code

- 🔧 [__init__.py](#--init---py) *(4 lines)*
- 🔧 [agent.py](#agent-py) *(664 lines)*
- 🔧 [client.py](#client-py) *(248 lines)*
- 🔧 [utils/console_formatter.py](#utils-console-formatter-py) *(94 lines)*
- 🔧 [utils/logger.py](#utils-logger-py) *(152 lines)*

**📊 Total Content:** 1,752 lines, 69,112 characters

---

# 📖 Documentation:

## 📖 Index

**File:** `docs/00_index.md`  
**Stats:** 47 lines, 1,500 characters

# Pocket-Agent Documentation

Welcome to the comprehensive documentation for the Pocket-Agent framework! This documentation is organized into focused sections to help you find the information you need quickly.

## 📚 Documentation Sections

### [Getting Started](01_getting-started.md)
- Quick start guide for creating your first Pocket-Agent
- Basic examples and configuration
- Essential concepts for beginners

### [Core Concepts](02_core-concepts.md)
- PocketAgent base class and parameters
- The step method and execution flow
- Message management and conversation history

### [Hooks and Events](03_hooks-and-events.md)
- Hook system for customizing agent behavior
- Event system for frontend integration
- Custom hook implementations

### [Multi-Model Support](04_multi-model-support.md)
- Working with different LLM providers
- LiteLLM Router integration
- Model configuration and switching

### [PocketAgentClient](05_client.md)
- Client wrapper for MCP protocol
- Custom query parameters
- Tool error handling and custom parsers
- Server-initiated events

### [Multi-Agent Systems](06_multi-agent.md)
- Building complex multi-agent architectures
- Sub-agent tool integration
- Orchestrating multiple specialized agents

### [Agent-as-a-Server](07_agent-as-a-server.md)
- Using Pocket-Agent as an MCP server
- Integration with external frameworks
- Server setup and configuration

### [Testing](08_testing.md)
- Test suite overview
- Running tests with different options
- Coverage reporting


---

## 📖 Getting Started

**File:** `docs/01_getting-started.md`  
**Stats:** 64 lines, 2,049 characters

# Getting Started with Pocket-Agent

## Out of the Box Behavior

You can use PocketAgent's default behavior in 3 steps:

1. Configure the agent:
    ```python
    from pocket_agent import AgentConfig

    # Configure and run
    config = AgentConfig(
        llm_model="gpt-4",
        system_prompt="You are a helpful assistant."
    )

2. Define the [JSON MCP config](https://gofastmcp.com/integrations/mcp-json-configuration) to add mcp servers which the agent can access:
    ```python
    servers = {
        "mcpServers": {
            "weather_tools": {
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
                "cwd": "path/to/server.py
            }
        }
    }
    ```

3. Initialize and run the agent:
    ```python
    import asyncio
    from pocket_agent import PocketAgent

    agent = PocketAgent(agent_config=config, mcp_config=servers)

    agent_final_answer = await agent.run("What's the weather fo London?")
    ```

In the above example, the `run` method will add the input as a user message, and then run in a loop generating new messages and running any tools called by the configured LLM until the configured LLM generates a response which does not have any tool calls.


## Creating Your First Custom Pocket-Agent (Quick Start)

The `run` method is intended to be reimplemented or extended for more complex use cases. For example, we can add a method to continually run the agent on user input until the user inputs `"quit"`:

```python
class SimpleAgent(PocketAgent):
    async def execute_user_input_loop(self):
        while True:
            user_input = input("Your input ('quit' to stop): ")
            if user_input.lower() == 'quit':
                break

            agent_result = await self.run(user_input)
```

This agent can be initialized in the following way (using the AgentConfig and MCP config from above):
```python
    agent = PocketAgent(agent_config=config, mcp_config=servers)
    await agent.execute_user_input_loop()
```


---

## 📖 Core Concepts

**File:** `docs/02_core-concepts.md`  
**Stats:** 90 lines, 3,264 characters

# Core Concepts

## 🏗️ **PocketAgent Base Class**

The `PocketAgent` is an abstract base class that provides the foundation for building custom agents. You inherit from this class and implement the `run()` method to define your agent's behavior.

```python
class MyAgent(PocketAgent):
    async def run(self, input: str) -> Union[FastMCPToolResult, dict, str]:
        # Your agent logic here
```

### PocketAgent Parameters:
```python
agent = PocketAgent(
    agent_config,   # Required: (AgentConfig) Instance of the AgentConfig class
    mcp_config,     # Optional: (dict or FastMCP) MCP Server or JSON MCP server configuration to pass tools to the agent
    router,         # Optional: A LiteLLM router instance to manage llm rate limits
    logger,         # Optional: A logger instance to capture logs
    hooks,          # Optional: (AgentHooks) optionally define custom behavior at common junction points
    sub_agents      # Optional: (list[PocketAgent]) list of Pocket-Agent's to be used as sub_agents
    **client_kwargs # Optional: additional kwargs passed to the PocketAgentClient

)
```

### AgentConfig Parameters:

```python
config = AgentConfig(
    llm_model="gpt-4",                    # Required: LLM model to use
    system_prompt="You are helpful...",   # Optional: System prompt for the agent
    agent_id="my-agent-123",              # Optional: Custom context ID
    allow_images=False,                   # Optional: Enable image input support (default: False)
    messages=[],                          # Optional: Initial conversation history (default: [])
    completion_kwargs={                   # Optional: Additional LLM parameters (default: {"tool_choice": "auto"})
        "tool_choice": "auto",
        "temperature": 0.7
    }
)
```

## 🔄 **The Step Method**

The `step()` method is the core execution unit that:
1. Gets an LLM response with available tools
2. Executes any tool calls in parallel
3. Updates conversation history

The output of calling the `step()` method is the StepResult
```python
@dataclass
class StepResult:
    llm_message: LitellmMessage                                 # The message generated by the llm including str content, tool calls, images, etc.
    tool_execution_results: Optional[list[ToolResult]] = None   # Results of any executed tools 
```

```python
# Single step execution
step_result = await agent.step()

# continue until no more tool calls
while step_result.llm_message.tool_calls is not None:
    step_result = await agent.step()
```

### Step Result Structure:
```python
{
    "llm_message": LitellmMessage,           # The LLM response
    "tool_execution_results": [ToolResult]   # Results from tool calls (if any)
}
```

## 💬 **Message Management**

Pocket Agent automatically adds llm generated messages and tool result messages in the `step()` function.
Input provided by a user can easily be managed using `add_user_message()` and should be done before calling the `step()` method:

```python
class Agent(PocketAgent)
    async def run(self):
        # Add user messages (with optional images)
        await agent.add_user_message("Hello!", image_base64s=["base64_image_data"])
        await self.step()

# Clear all messages except the system prompt
agent.reset_messages()
```


---

## 📖 Hooks And Events

**File:** `docs/03_hooks-and-events.md`  
**Stats:** 95 lines, 3,351 characters

# Hooks and Events

## 🪝 **Hook System**

Customize agent behavior at key execution points:

```python
@dataclass
class HookContext:
    """Context object passed to all hooks"""
    agent: 'PocketAgent'                                    # provides hooks access to the Agent instance
    metadata: Dict[str, Any] = field(default_factory=dict)  # additional metadata (default is empty)

class CustomHooks(AgentHooks):
    async def pre_step(self, context: HookContext):
        # executed before the llm response is generated in the step() method
        print("About to execute step")
    
    async def post_step(self, context: HookContext):
        # executed after all tool results (if any) are retrieved; This runs even if tool calling results in an error
        print("Step completed")
    
    async def pre_tool_call(self, context: HookContext, tool_call):
        # executed right before a tool is run
        print(f"Calling tool: {tool_call.name}")
        # Return modified tool_call or None
    
    async def post_tool_call(self, context: HookContext, tool_call, result):
        # executed right after a tool call result is retrieved from the PocketAgentClient
        print(f"Tool {tool_call.name} completed")
        return result  # Return modified result
    
    async def on_llm_response(self, context: HookContext, response):
        # executed right after a response message has been generated by the llm
        print("Got LLM response")
    
    async def on_event(self, event: AgentEvent):
        # Custom publishing of events useful for frontend integration

    async def on_tool_error(self, context: HookContext, tool_call: MCPCallToolRequestParams, error: Exception) -> Union[str, False]:
        # custom error handling described in more detail in PocketAgentClient docs 

    ####
    # PocketAgent uses a default result parser which will be replaced by this implementation if implemented
    ####
    async def on_tool_result(self, context: HookContext, tool_call: ChatCompletionMessageToolCall, tool_result: FastMCPCallToolResult) -> ToolResult:
        # custom parser for tool results described in more detail in PocketAgentClient docs

# Use custom hooks
agent = MyAgent(
    agent_config=config,
    mcp_config=mcp_config,
    hooks=CustomHooks()
)
```

By Default, the HookContext is created with the Agent instance and empty metadata but this behavior can be customized by implementing the `_create_hook_context` method in your custom agent:
```python
class Agent(PocketAgent):
    async def _create_hook_context(self) -> HookContext:
        return HookContext(
            agent=self,
            metadata={
                # custom metadata
            }
        )

```

## 📡 **Event System**

PocketAgent includes an AgentEvent type:

```python
@dataclass
class AgentEvent:
    event_type: str  # e.g., "new_message"
    data: dict       # Event-specific data
```

By default, events are automatically emitted when any new message is added to the message history:
- llm message
- tool result message
- user message

You can easily add `on_event` calls with custom AgentEvents in other hooks if necessary:
```python
class CustomHooks(AgentHooks):
    async def pre_tool_call(self, context, tool_call):
        event = AgentEvent(
            event_type="tool_call",
            data=tool_call
        )
```


---

## 📖 Multi Model Support

**File:** `docs/04_multi-model-support.md`  
**Stats:** 46 lines, 1,008 characters

# Multi-Model Support

## 🔧 **Multi-Model Support**

Works seamlessly with any LiteLLM-supported model:

```python
# OpenAI
config = AgentConfig(llm_model="gpt-4")

# Anthropic
config = AgentConfig(llm_model="anthropic/claude-3-sonnet-20240229")

# Local models
config = AgentConfig(llm_model="ollama/llama2")

# Azure OpenAI
config = AgentConfig(llm_model="azure/gpt-4")
```

## 🚏 **LiteLLM Router Integration**
To easily set rate limits or implement load balancing with multiple LLM API providers you can pass a [LiteLLM Router](https://docs.litellm.ai/docs/routing) instance to PocketAgent:

```python
from litellm import Router
router_info = {
    "models": [
        {
            "model_name": "gpt-5-nano",
            "litellm_params": {
                "model": "gpt-5-nano",
                "tpm": 3000000,
                "rpm": 5000
            }
        }
    ]
}

litellm_router = Router(model_list=router_info["models"])

agent = PocketAgent(
    router=litellm_router,
    # other args
)
```


---

## 📖 Client

**File:** `docs/05_client.md`  
**Stats:** 115 lines, 6,004 characters

# PocketAgentClient

Each PocketAgent instance initializes a PocketAgentClient which acts as a wrapper for the [FastMCP Client](https://gofastmcp.com/clients/client) to implement the standard mcp protocol features and some additional features.

## Custom Metadata

- Sending custom metadata is now supported by the MCP protocol. As such, Pocket Agent also supports sending custom metadata to MCP servers during tool calls. By default, Pocket Agent sends the following metadata on each tool call:
    ```python
    {
        "tool_call_id": tool_call.id,
        "context_id": self.agent_id,
        "agent_name": self.agent_config.name,
        "is_sub_agent": self.is_sub_agent
    }
    ```
- To add additional custom values to the metadata, the `PocketAgent` class exposes a  `custom_tool_call_metadata` parameter which accepts a `dict`. Custom metadata will be merged with the default metadata. For example:
    ```python
    # Create a PocketAgent instance with custom metadata
    agent = PocketAgent(
        ...
        custom_tool_call_metadata = {
            "context_id": "1111",
            "agent_category": "researcher"
        }
    )

    # Every tool call from the agent will have its `meta` set to this:
    {
        "tool_call_id": tool_call.id,
        "context_id": "1111",
        "agent_name": self.agent_config.name,
        "is_sub_agent": self.is_sub_agent,
        "agent_category": "researcher"
    }
    ```
- In order to access tool call metadata from within an MCP tool you can simply use the automatically injected Context:
    ```python
    BASE_PATHS = {
        "1432": "home/dir/1432"
        "1111": "home/dir/1111"
    }
    @mcp.tool
    def save_file(file_name: str, file_content: str, ctx: Context) -> str:
        meta = ctx.request_context.meta
        base_path = "/dir/0"
        if meta:
            context_id = meta.context_id if hasattr(meta, 'context_id') else None
            if context_id:
                base_path = BASE_PATHS.get(context_id, base_path)
        file_path = os.path.join(base_path, file_name)
        # Save file to path

        return f"File saved"
    ```

## on_tool_error (hook)

- If a tool call fails and is not handled within the tool itself, it will result in a ToolError result. You can add custom handling of such errors using the `on_tool_error` hook method in `AgentHooks`. Any custom functionality should either return a `string` or `False`. If the method returns a `string`, the contents will be sent to the agent as the tool result, if the method returns `False` the ToolError will be raised an execution of the agent will stop.
The following handler is implemented by default to handle a common scenario where LLMs pass invalid parameters to tools resulting in an error:

    ```python
    class AgentHooks:
        async def on_tool_error(self, context: HookContext, tool_call: MCPCallToolRequestParams, error: Exception) -> Union[str, False]:
            if "unexpected_keyword_argument" in str(error):
                tool_call_name = tool_call.name
                tool_format = await context.agent.mcp_client.get_tool_input_format(tool_call_name)
                return "You supplied an unexpected keyword argument to the tool. \
                    Try again with the correct arguments as specified in expected format: \n" + tool_format
            return False
    ```

## tool_result_handler (hook)

- When a tool is called successfully it results in a [CallToolResult](https://github.com/jlowin/fastmcp/blob/09ae8f5cfdc62e6ac0bb7e6cc8ade621e9ebbf3e/src/fastmcp/client/client.py#L935). Most of the time, you will likely just want to parse the content which is a list of MCP content objects (i.e. [TextContent, ImageContent, etc](https://github.com/modelcontextprotocol/python-sdk/blob/c3717e7ad333a234c11d084c047804999a88706e/src/mcp/types.py#L662)). For this reason, the `PocketAgentClient` uses its default parser to parse these objects into content that can directly be fed to the agent as a message. Specifically, the default parser will return a ToolResult object:

    ```python
    return ToolResult(
        tool_call_id=tool_call.id,                              # ID of the original tool call (needed by most apis when passing tool results)
        tool_call_name=tool_call.name,                          # Name of the tool which the result is for
        tool_result_content=tool_result_content,                # Tool result content compatible with LiteLLM message format
        _extra={
            "tool_result_raw_content": tool_result_raw_content  # Unprocessed MCP tool result (unused by default)
        }
    )
    ```

    However, in some cases you may want to specifically parse structured content from a known tool in which case you can override the default parser by implementing the `on_tool_result` hook method in `AgentHooks`:

    ```python
    class CustomHooks(AgentHooks):
        async def on_tool_result(self, context: HookContext, tool_call: ChatCompletionMessageToolCall, tool_result: FastMCPCallToolResult) -> ToolResult:
            # your custom tool result parsing
    ```

## Server-initiated Events
- The MCP protocol implements numerous server-initiated events which should be handled by MCP clients. Each of these are documented here:
     - [Elicitation](https://gofastmcp.com/clients/elicitation)
     - [Logging](https://gofastmcp.com/clients/logging)
     - [Progress](https://gofastmcp.com/clients/progress)
     - [Sampling](https://gofastmcp.com/clients/sampling)
     - [Messages](https://gofastmcp.com/clients/messages)

By default, PocketAgent only implements the logging handler.

To define custom behavior for any other server initiated events they can be passed as additional arguments to the agent:
```python
agent = PocketAgent(
    elicitation_handler=your_elicitation_handler,
    log_handler=your_log_handler,
    progress_handler=your_progress_handler,
    sampling_handler=your_sampling_handler,
    message_handler=your_message_handler
)
```


---

## 📖 Multi Agent

**File:** `docs/06_multi-agent.md`  
**Stats:** 92 lines, 3,730 characters

# Multi-Agent Systems

PocketAgent natively supports multi-agent architectures where you can compose agents by passing other PocketAgent instances as sub-agents. Sub-agents are automatically converted to MCP tools that the main agent can call.

## Basic Multi-Agent Setup

```python
from pocket_agent import PocketAgent, AgentConfig

# Create a sub-agent with specialized capabilities
sub_agent_config = AgentConfig(
    llm_model="gpt-3.5-turbo",
    name="MathAgent", 
    role_description="A specialized agent for mathematical calculations",
    system_prompt="You are an expert mathematician. Solve mathematical problems step by step."
)

math_agent = PocketAgent(
    agent_config=sub_agent_config,
    mcp_config=math_mcp_config  # MCP config with math tools
)

# Create main agent with the sub-agent
main_config = AgentConfig(
    llm_model="gpt-4",
    name="MainAgent",
    system_prompt="You are a helpful assistant. Use the MathAgent when you need to solve math problems."
)

main_agent = PocketAgent(
    agent_config=main_config,
    mcp_config=main_mcp_config,  # Optional: main agent can have its own tools too
    sub_agents=[math_agent]      # Pass sub-agents as a list
)
```

## Multiple Sub-Agents

You can create complex multi-agent systems with multiple specialized sub-agents:

```python
# Create specialized sub-agents
research_agent = PocketAgent(
    agent_config=AgentConfig(
        llm_model="gpt-3.5-turbo",
        name="ResearchAgent",
        role_description="Specialized in web research and information gathering",
        system_prompt="You are a research specialist. Find and analyze information from web sources."
    ),
    mcp_config=research_mcp_config
)

analysis_agent = PocketAgent(
    agent_config=AgentConfig(
        llm_model="gpt-3.5-turbo", 
        name="AnalysisAgent",
        role_description="Specialized in data analysis and visualization",
        system_prompt="You are a data analyst. Analyze data and create visualizations."
    ),
    mcp_config=analysis_mcp_config
)

# Main orchestrator agent with multiple sub-agents
orchestrator = PocketAgent(
    agent_config=AgentConfig(
        llm_model="gpt-4",
        name="Orchestrator",
        system_prompt="You coordinate between specialized agents to complete complex tasks."
    ),
    sub_agents=[research_agent, analysis_agent],
    mcp_config=None            # mcp_config can be none if sub_agents are being used and the main agent doesn't need its own tools
)
```

## Sub-Agent Tool Integration

When you add sub-agents to a main agent, they are automatically exposed as tools with names formatted as `{agent_name}-message` with a single `message: str` argument. The main agent can call these tools to interact with sub-agents.

When a sub-agent tool is called, it will execute the agent's `run` method with the `message` tool call argument.
Therefore, **the `run` method of a sub-agent must accept one argument as an input message and return.**
Additionally, the `run` method must return either `None` or one of these types:
 - `str`
 - `dict`
 - Instance of [FastMCP's ToolResult](https://github.com/jlowin/fastmcp/blob/39a1e59bfd9a665fd961d18418c5addde94c3019/src/fastmcp/tools/tool.py#L66C7-L66C17)


## Sub-agent Execution Lock

In some cases the primary agent may invoke parallel calls to the same sub-agent. To avoid unexpected behavior in these scenarios, the sub-agent's `run` method is executed within a lock (i.e. only one invocation of a single agent's `run` method can execute at a time).

If the sub-agent is able to handle multiple tasks at once, a simple workaround to avoid bottlenecks due to synchronous execution of parallel tool calls to a sub-agent is to instruct the parent agent to combine


---

## 📖 Agent As A Server

**File:** `docs/07_agent-as-a-server.md`  
**Stats:** 19 lines, 705 characters

# Pocket Agent as an MCP Server

Any Pocket Agent instance can be used as a standalone MCP server to be integrated with external frameworks.

This example shows how to set up a Pocket Agent as a FastMCP server:

```python
agent = PocketAgent(
    agent_config= # AgentConfig,
    mcp_config= # MCP server config
    )

mcp_server = agent.as_mcp_server() # returns an instance of FastMCP
```

The MCP server generated in the above example will have a single `{agent_name}-message` tool. See [Multi-Agent Systems - Sub-Agent Tool Integration](multi-agent.md#sub-agent-tool-integration) for more details on the `message` tool as it is the same.

*Note: Even agents with sub-agents can be run as MCP Servers*


---

## 📖 Testing

**File:** `docs/08_testing.md`  
**Stats:** 22 lines, 567 characters

# Testing

Pocket Agent includes a comprehensive test suite covering all core functionality. The tests are designed to be fast and reliable using in-memory FastMCP servers and mocked LLM responses.

## Running Tests

The easiest way to run tests is using the provided test runner script:

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run with coverage reporting (Coverage reports are generated in `htmlcov/`)
python run_tests.py --coverage

# Run quick subset for development
python run_tests.py --quick
```


---

# 🔧 Source Code:

## 🔧 __init__.py

**File Path:** `__init__.py`  
**Stats:** 4 lines, 244 characters

*Package initialization and public API exports*

```python
from .agent import PocketAgent, AgentConfig, AgentEvent, AgentHooks
from .client import PocketAgentClient, PocketAgentToolResult

__all__ = ["PocketAgent", "AgentConfig", "AgentEvent", "PocketAgentClient", "PocketAgentToolResult", "AgentHooks"]
```

---

## 🔧 agent.py

**File Path:** `agent.py`  
**Stats:** 664 lines, 26,661 characters

*Core PocketAgent class - the heart of the framework with agent lifecycle management*

```python
from litellm import Router
import litellm
from litellm.types.utils import (
    ChatCompletionMessageToolCall, 
    Message as LitellmMessage, 
    ModelResponse as LitellmModelResponse
)
from mcp.types import (
    CallToolRequestParams as MCPCallToolRequestParams,
    TextContent as MCPTextContent,
    ImageContent as MCPImageContent,
)
from abc import abstractmethod
from fastmcp.client.logging import LogMessage
from fastmcp.tools.tool import ToolResult as FastMCPToolResult, Tool as FastMCPTool
from fastmcp import FastMCP
from pydantic import Field
import uuid
import asyncio  # Add this import
from typing import Optional, Tuple

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Union
import json
import traceback


from pocket_agent.client import PocketAgentClient, PocketAgentToolResult
from pocket_agent.utils.logger import configure_logger as configure_pocket_agent_logger


@dataclass
class AgentEvent:
    event_type: str
    data: dict
    meta: Optional[dict] = field(default_factory=dict)

@dataclass
class StepResult:
    llm_message: LitellmMessage
    tool_execution_results: Optional[list[PocketAgentToolResult]] = None


@dataclass
class AgentConfig:
    """Configuration class to make agent setup cleaner"""
    llm_model: str
    name: Optional[str] = None
    role_description: Optional[str] = None
    agent_id: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: Optional[list[dict]] = None
    allow_images: Optional[bool] = False
    completion_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "tool_choice": "auto"
        # we can add more llm config items if needed
    })

    def get_completion_kwargs(self) -> Dict[str, Any]:
        # safely get completion kwargs
        return self.completion_kwargs or {}


@dataclass
class HookContext:
    """Context object passed to all hooks"""
    agent: 'PocketAgent'
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentHooks:
    """Centralized hook registry with consistent signatures"""
    
    async def pre_step(self, context: HookContext) -> None:
        pass
    
    async def post_step(self, context: HookContext) -> None:
        pass
    
    async def pre_tool_call(self, context: HookContext, tool_call: MCPCallToolRequestParams) -> Optional[MCPCallToolRequestParams]:
        return None
    
    async def post_tool_call(self, context: HookContext, tool_call: MCPCallToolRequestParams, result: PocketAgentToolResult) -> Optional[PocketAgentToolResult]:
        return result  # Return modified or None to indicate error

    async def on_llm_response(self, context: HookContext, response: LitellmModelResponse) -> None:
        pass
    
    async def on_event(self, event: AgentEvent) -> None:
        from pocket_agent.utils.console_formatter import ConsoleFormatter
        formatter = ConsoleFormatter()
        formatter.format_event(event)

    #########################################################
    # The following 2 hooks run in the PocketAgentClient
    #########################################################

    # tool result handler will replace the default tool result handler in PocketAgentClient if implemented
    # async def on_tool_result(self, context: HookContext, tool_call: ChatCompletionMessageToolCall, tool_result: FastMCPCallToolResult) -> PocketAgentToolResult:
    #     pass
    
    async def on_tool_error(self, context: HookContext, tool_call: MCPCallToolRequestParams, error: Exception) -> Union[str, False]:
        if "unexpected_keyword_argument" in str(error):
            tool_call_name = tool_call.name
            tool_format = await context.agent.mcp_client.get_tool_input_format(tool_call_name)
            return "You supplied an unexpected keyword argument to the tool. \
                Try again with the correct arguments as specified in expected format: \n" + json.dumps(tool_format)
        return False



class PocketAgent:
    def __init__(self,
                 agent_config: AgentConfig,
                 mcp_config: Optional[Union[dict, FastMCP]] = None,
                 router: Optional[Router] = None,
                 logger: Optional[logging.Logger] = None,
                 hooks: Optional[AgentHooks] = None,
                 sub_agents: Optional[list["PocketAgent"]] = None,
                 custom_tool_call_metadata: Optional[dict] = None,
                 **client_kwargs):

        self.logger = logger or configure_pocket_agent_logger()
        self.agent_config = agent_config

        if not self.agent_config.agent_id:
            self.agent_config.agent_id = str(uuid.uuid4())

        if not agent_config.name:
            self.logger.debug("Agent name is not set, using agent_id as name")
            self.agent_config.name = self.agent_config.agent_id

        if router:
            self.llm_completion_handler = router
        else:
            self.llm_completion_handler = litellm
        self.agent_config = agent_config
        self.hooks = hooks or AgentHooks()

        self._run_lock = asyncio.Lock()

        if not mcp_config and not sub_agents:
            self.logger.debug("MCP config is empty and no sub agents are provided. \
                The agent will be initialized with no tools.")
            mcp_config = FastMCP()
        
        self.mcp_config = mcp_config
        self.sub_agents = sub_agents

        # Mark sub agents as sub agents
        if self.sub_agents:
            for sub_agent in self.sub_agents:
                sub_agent.is_sub_agent = True

        self.custom_tool_call_metadata = custom_tool_call_metadata or {}
            
        self.system_prompt = agent_config.system_prompt or ""
        self.messages = agent_config.messages or []

        self.logger.info(
            f"Initializing PocketAgent '{self.name}'",
            extra={
                'agent_id': self.agent_id,
                'model': self.agent_config.llm_model,
                'has_system_prompt': bool(self.system_prompt),
                'initial_messages': len(self.messages),
                'allow_images': self.allow_images,
                'has_sub_agents': self.has_sub_agents,
                'sub_agent_count': self.sub_agent_count
            }
        )

        self._init_client(self.mcp_config, self.sub_agents, **client_kwargs)
        
        # After successful initialization:
        self.logger.info(
            f"PocketAgent '{self.name}' ready - {len(self.messages)} messages, "
            f"{'with' if self.has_sub_agents else 'no'} sub-agents"
        )


    #########################################################
    # Properties
    #########################################################
    @property
    def model(self) -> str:
        return self.agent_config.llm_model

    @property
    def name(self) -> str:
        """Agent name"""
        return self.agent_config.name

    @property  
    def agent_id(self) -> str:
        """Unique agent identifier"""
        return self.agent_config.agent_id

    @property
    def role_description(self) -> Optional[str]:
        """Description of the agent's role/purpose"""
        return self.agent_config.role_description

    @property
    def completion_kwargs(self) -> Dict[str, Any]:
        """LLM completion parameters"""
        return self.agent_config.get_completion_kwargs()

    @property
    def message_count(self) -> int:
        """Number of messages in conversation history"""
        return len(self.messages)

    @property
    def has_sub_agents(self) -> bool:
        """Whether this agent has sub-agents"""
        if self.sub_agents is None:
            return False
        return len(self.sub_agents) > 0

    @property
    def sub_agent_count(self) -> int:
        """Number of sub-agents"""
        if self.sub_agents is None:
            return 0
        return len(self.sub_agents)

    @property
    def is_sub_agent(self) -> bool:
        """Whether this agent is a sub-agent"""
        return getattr(self, '_is_sub_agent', False)
    
    @is_sub_agent.setter
    def is_sub_agent(self, value: bool) -> None:
        self._is_sub_agent = value

    @property
    def is_configured(self) -> bool:
        """Whether the agent is properly configured"""
        return bool(self.agent_config.llm_model and self.mcp_client)

    @property
    def allow_images(self) -> bool:
        return self.agent_config.allow_images



    def _init_client(self, mcp_config: Union[dict, FastMCP, None], sub_agents: Optional[list["PocketAgent"]] = None, **client_kwargs) -> None:
        """
        Initialize the most basic MCP client with the given configuration.
        Override this to add custom client handlers.
        """
        # check if on_tool_result is in hooks class
        if hasattr(self.hooks, "on_tool_result"):
            tool_result_handler = self.create_wrapper_with_context(self.hooks.on_tool_result)
        else:
            tool_result_handler = None

        if isinstance(mcp_config, dict):
            self.mcp_client = PocketAgentClient(mcp_config=mcp_config,
                        on_tool_error=self.create_wrapper_with_context(self.hooks.on_tool_error),
                        tool_result_handler=tool_result_handler,
                        **client_kwargs)
            if sub_agents:
                composed_sub_agents_server = self._create_composed_sub_agents_server()
                self._mount_server_to_mcp_client(composed_sub_agents_server)
        elif isinstance(mcp_config, FastMCP):
            if sub_agents:
                composed_sub_agents_server = self._create_composed_sub_agents_server()
                mcp_config.mount(composed_sub_agents_server)
            self.mcp_client = PocketAgentClient(mcp_config=mcp_config,
                        on_tool_error=self.create_wrapper_with_context(self.hooks.on_tool_error),
                        tool_result_handler=tool_result_handler,
                        **client_kwargs)
        elif sub_agents:
            composed_sub_agents_server = self._create_composed_sub_agents_server()
            self.mcp_client = PocketAgentClient(mcp_config=composed_sub_agents_server,
                        on_tool_error=self.create_wrapper_with_context(self.hooks.on_tool_error),
                        tool_result_handler=tool_result_handler,
                        **client_kwargs)
        else:
            raise ValueError(f"Failed to initialize mcp client: expected mcp config to be of type \
                dict, FastMCP, or None, but got {type(mcp_config)} and/or sub agents to be of type list, but got {type(sub_agents)}.\
                both cannot be None or invalid types.")


                        
    def create_wrapper_with_context(self, func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            hook_context = self._create_hook_context()
            return await func(hook_context, *args, **kwargs)
        return wrapper


    def _format_messages(self) -> list[dict]:
        # format system prompt and messages in proper format
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.messages
        ]
        self.logger.debug(f"Formatted messages: {messages}")
        return messages



    async def _get_llm_response(self, **override_completion_kwargs) -> LitellmModelResponse:
        self.logger.debug(f"Requesting LLM response with model={self.agent_config.llm_model}, message_count={len(self.messages) + 1}")
        # get a response from the llm
        kwargs = self.agent_config.get_completion_kwargs()
        kwargs.update(override_completion_kwargs)
        messages = self._format_messages()
        tools = await self.mcp_client.get_tools(format="openai")
        if not tools:
            self.logger.debug("No tools found, LLM response will be generated without tools")
            # remove tool_choice from kwargs if it exists
            if "tool_choice" in kwargs:
                del kwargs["tool_choice"]
        else:
            kwargs.update({
                "tools": tools,
            })
        try:
            self.logger.debug(f"Requesting LLM response with kwargs={kwargs}")
            response = await self.llm_completion_handler.acompletion(
                model=self.agent_config.llm_model,
                messages=messages,
                **kwargs
            )
            
            self.logger.debug(f"LLM response received: full_response={response}")
            return response
        except Exception as e:
            self.logger.error(
                f"LLM request failed - Model: {self.agent_config.llm_model}, "
                f"Messages: {len(messages)}, Tools: {len(tools) if tools else 0}",
                exc_info=True,
                extra={
                    'agent_id': self.agent_id,
                    'model': self.agent_config.llm_model,
                    'message_count': len(messages)
                }
            )
            raise
    


    async def add_message(self, message: dict, **meta) -> None:
        self.logger.debug(f"Adding message: {message}")
        meta.update({"agent_name": self.agent_config.name, "is_sub_agent": self.is_sub_agent})
        event = AgentEvent(event_type="new_message", data=message, meta=meta)
        await self.hooks.on_event(event)
        self.messages.append(message)

    
    async def add_llm_message(self, llm_message: LitellmMessage) -> None:
        await self.add_message(llm_message.model_dump())

    async def add_tool_result_message(self, tool_result_message: dict) -> None:
        await self.add_message(tool_result_message)


    def _filter_images_from_tool_result_content(self, tool_result_content: list[dict]) -> list[dict]:
        return [content for content in tool_result_content if content["type"] != "image_url"]



    async def _call_single_tool_with_hooks(self, tool_call: MCPCallToolRequestParams) -> PocketAgentToolResult:
        """Execute a single tool call with hooks."""
        hook_context = self._create_hook_context()
    
        transformed_tool_call = await self.hooks.pre_tool_call(hook_context, tool_call)
        if transformed_tool_call is not None:
            tool_call = transformed_tool_call
        try:
            result = await self.mcp_client.call_tool(tool_call)
            transformed_result = await self.hooks.post_tool_call(hook_context, tool_call, result)
            if transformed_result is not None:
                result = transformed_result
            return result
        except Exception as e:
            self.logger.error(
                f"Tool call '{tool_call.name}' failed and was not handled by tool error hook",
                exc_info=True,
                extra={
                    'agent_id': self.agent_id,
                    'tool_name': tool_call.name,
                    'tool_arguments': str(tool_call.arguments)[:200],  # Limit length
                    'error_type': type(e).__name__
                }
            )
            raise
    
    def add_metadata_to_tool_calls(self, tool_calls: list[MCPCallToolRequestParams]) -> list[MCPCallToolRequestParams]:
        """Add metadata to tool calls"""
        self.logger.debug(f"Adding metadata to tool calls: {tool_calls}")
        for tool_call in tool_calls:
            tool_call_metadata = {
                "tool_call_id": tool_call.id,
                "context_id": self.agent_id,
                "agent_name": self.agent_config.name,
                "is_sub_agent": self.is_sub_agent
                }
            tool_call_metadata.update(self.custom_tool_call_metadata)
            tool_call.meta = tool_call_metadata
        self.logger.debug(f"Tool calls with metadata: {tool_calls}")
        return tool_calls


    async def _call_tools(self, tool_calls: list[ChatCompletionMessageToolCall]) -> list[PocketAgentToolResult]:
        """Execute all tool calls in parallel, with individual hooks for each."""
        mcp_tool_calls = [self.mcp_client.transform_tool_call_request(tool_call) for tool_call in tool_calls]
        mcp_tool_calls = self.add_metadata_to_tool_calls(mcp_tool_calls)
        tool_results = await asyncio.gather(*[
            self._call_single_tool_with_hooks(tool_call) 
            for tool_call in mcp_tool_calls
        ])
        return tool_results


    async def step(self, **override_completion_kwargs) -> dict:
        """
        Execute a single step of the agent.
        A step includes a single LLM response and execution of any tool calls that were found in the LLM response.
        """
        self.logger.debug("Starting agent step")
        hook_context = self._create_hook_context()
        await self.hooks.pre_step(hook_context)
        
        step_result = None
        step_phase = "llm_response"
        try:
            llm_response = await self._get_llm_response(**override_completion_kwargs)
            await self.hooks.on_llm_response(hook_context, llm_response)
            llm_message = llm_response.choices[0].message
            await self.add_llm_message(llm_message)
            if llm_message.tool_calls:
                step_phase = "tool_execution"
                tool_details = [
                    f"{tool_call.function.name}"
                    for tool_call in llm_message.tool_calls
                ]
                self.logger.info(
                    f"Executing {len(llm_message.tool_calls)} tool calls: {', '.join(tool_details)}",
                    extra={
                        'agent_id': self.agent_id,
                        'tool_count': len(llm_message.tool_calls),
                        'tool_details': tool_details
                    }
                )
                tool_execution_results = await self._call_tools(llm_message.tool_calls)
                if tool_execution_results:
                    step_phase = "tool_execution_result_processing"
                    self.logger.debug(f"Received tool execution results: {tool_execution_results}")
                    for tool_execution_result in tool_execution_results:
                        tool_call_id = tool_execution_result.tool_call_id
                        tool_call_name = tool_execution_result.tool_call_name
                        tool_result_content = tool_execution_result.tool_result_content
                        if not self.allow_images:
                            self.logger.debug("allow images set to false, filtering images from tool result content")
                            tool_result_content = self._filter_images_from_tool_result_content(tool_result_content)
                        new_message = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_call_name,
                            "content": tool_result_content
                        }
                        await self.add_tool_result_message(new_message)
                        step_result = StepResult(llm_message=llm_message, tool_execution_results=tool_execution_results)
                else:
                    self.logger.error("No tool execution results received")
                    raise ValueError("No tool execution results received. Tool calls must have failed silently.")
            else:
                self.logger.debug("No tool calls in generate result")
                step_result = StepResult(llm_message=llm_message)
        except Exception as e:
            self.logger.error(f"Error in step: {step_phase} phase", exc_info=True, extra={
                'agent_id': self.agent_id,
                'step_phase': step_phase,
                'message_count': len(self.messages),
                'has_tool_calls': 'llm_message' in locals() and bool(llm_message.tool_calls)
            })
            raise
        finally:
            # Post-step hook
            await self.hooks.post_step(hook_context)
            
            if step_result is None:
                self.logger.debug("Step result is None")
            return step_result

    

    async def add_user_message(self, user_message: str, image_base64s: Optional[list[str]] = None) -> None:
        image_count = len(image_base64s) if image_base64s else 0
        self.logger.info(f"Adding user message: {user_message} with {image_count} images")
        new_message_content = [
            {
                "type": "text",
                "text": user_message
            }
        ]
        if not self.allow_images:
            if image_base64s:
                self.logger.warning("allow images set to false, but images were provided, ignoring images")
            image_base64s = None
            
        else:
            if image_base64s:
                for image_base64 in image_base64s:
                    new_message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    })
        await self.add_message({
            "role": "user",
            "content": new_message_content
        })

    
    def reset_messages(self) -> None:
        self.messages = []


    async def _run_with_lock(self, *args, **kwargs) -> Union[FastMCPToolResult, dict, str]:
        """
        Thread-safe wrapper around run() method
        """
        async with self._run_lock:
            self.logger.debug(f"Acquired run lock for agent {self.name}")
            try:
                return await self.run(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in run: {traceback.format_exc()}")
                raise
            finally:
                self.logger.debug(f"Released run lock for agent {self.name}")


    async def run(self, user_input: str, **kwargs) -> Union[FastMCPToolResult, dict, str]:
        """
        Run the agent.
        Returns the the final result as a FastMCPToolResult, dict, or str.
        """
        # add the user message to the message history
        await self.add_user_message(user_input)

        # generate llm response and run all tool calls (if any)
        step_result = await self.step()

        # continue until no more tool calls
        while step_result.llm_message.tool_calls is not None:
            step_result = await self.step()

        # return the last llm message without tool calls
        return step_result.llm_message.content


    def _create_hook_context(self) -> HookContext:
        """Create a hook context for the current state"""
        return HookContext(
            agent=self,
            metadata={}
        )


    def _format_composed_sub_agent_server_name(self) -> str:
        """Format the composed sub agent server name"""
        return f"interact_with_agent"

    def _format_sub_agent_tool_name(self, agent_name: str) -> str:
        """Format the sub agent tool name"""
        return f"{agent_name}-message"

    def _format_sub_agent_server_description(self, agent_name: str, agent_role_description: str) -> str:
        """Format the sub agents server description"""
        agent_tool_description = f"Send a message to the {agent_name} agent."
        if agent_role_description:
            agent_tool_description += f"Purpose of the {agent_name} agent: \n{agent_role_description}"
        else:
            self.logger.warning(f"Agent role description is not set, sub agent tool description only contains ({agent_tool_description})")
        return agent_tool_description


    def _mount_server_to_mcp_client(self, server: FastMCP) -> None:
        self.mcp_client.mount_server(server)


    def _create_composed_sub_agents_server(self) -> FastMCP:
        """Create a composed sub agents server"""
        server_name = self._format_composed_sub_agent_server_name()
        composed_sub_agents_server = FastMCP(name=server_name)
        for sub_agent in self.sub_agents:
            sub_agent_tool = sub_agent.as_mcp_tool()
            composed_sub_agents_server.add_tool(sub_agent_tool)
        return composed_sub_agents_server


    def as_mcp_tool(self) -> FastMCPTool:
        """
        Return the agent as an in-memory sub-agent server.
        """
        # get agent name and role description
        agent_name = self.agent_config.name
        agent_role_description = self.agent_config.role_description
        agent_tool_name = self._format_sub_agent_tool_name(agent_name)
        agent_tool_description = self._format_sub_agent_server_description(agent_name, agent_role_description)

        async def message_agent(message: str = Field(..., description=f"The message to send to the {agent_name} agent")) -> FastMCPToolResult:
            # run the agent
            result = await self._run_with_lock(message)
            if isinstance(result, FastMCPToolResult):
                return result
            elif isinstance(result, dict):
                return FastMCPToolResult(
                    content=[
                        MCPTextContent(
                            type="text",
                            text=json.dumps(result),
                            meta={"mime_type": "application/json"}
                        )
                    ]
                )
            elif isinstance(result, str):
                return FastMCPToolResult(
                    content=[
                        MCPTextContent(
                            type="text",
                            text=result,
                            meta={"mime_type": "text/plain"}
                        )
                    ]
                )
            elif result is None:
                return FastMCPToolResult(
                    content=[
                        MCPTextContent(
                            type="text",
                            text="No result returned",
                            meta={"mime_type": "text/plain"}
                        )
                    ]
                )
            else:
                raise ValueError(f"Unsupported result type: {type(result)}. Result must be a FastMCPToolResult, dict, or str.")

        return FastMCPTool.from_function(
            fn=message_agent,
            name=agent_tool_name,
            description=agent_tool_description
        )

    def as_mcp_server(self) -> FastMCP:
        """
        Return the agent as its own MCP server.
        """
        mcp_server = FastMCP(name=self.agent_config.name, instructions=self.agent_config.role_description)
        mcp_server.add_tool(self.as_mcp_tool())
        return mcp_server



```

---

## 🔧 client.py

**File Path:** `client.py`  
**Stats:** 248 lines, 10,538 characters

*PocketAgentClient for MCP server communication and tool execution*

```python
from fastmcp import Client as MCPClient, FastMCP as FastMCPServer
from fastmcp.client.client import CallToolResult as FastMCPCallToolResult
from fastmcp.exceptions import ToolError
from fastmcp.client.logging import LogMessage
from mcp.types import (
    ListToolsResult, 
    ListResourcesResult, 
    ListResourceTemplatesResult, 
    CallToolResult as MCPCallToolResult,
    CallToolRequestParams as MCPCallToolRequestParams
)
from litellm.experimental_mcp_client.tools import (
                transform_mcp_tool_to_openai_tool,
                transform_openai_tool_call_request_to_mcp_tool_call_request,
            )
from litellm.types.utils import ChatCompletionMessageToolCall
from dataclasses import dataclass
from typing import Optional, Callable, Literal, Union
import asyncio
import logging
import copy

from pocket_agent.utils.logger import configure_logger as configure_pocket_agent_logger



@dataclass
class PocketAgentToolResult:
    tool_call_id: str
    tool_call_name: str
    tool_result_content: list[dict]
    _extra: Optional[dict] = None


class PocketAgentClient:
    def __init__(self, mcp_config: Union[dict, FastMCPServer], 
                 mcp_logger: Optional[logging.Logger] = None,
                 log_handler: Optional[Callable[[LogMessage], None]] = None,
                 client_logger: Optional[logging.Logger] = None,
                 on_tool_error: Optional[Callable[[ChatCompletionMessageToolCall, Exception], bool]] = None,
                 mcp_server_query_params: Optional[dict] = None,
                 tool_result_handler: Optional[Callable[[ChatCompletionMessageToolCall, FastMCPCallToolResult], PocketAgentToolResult]] = None,
                 **kwargs
                 ):

        # Separate loggers for different purposes
        self.client_logger = client_logger or configure_pocket_agent_logger(name="pocket_agent.client")
        self.mcp_logger = mcp_logger or logging.getLogger("pocket_agent.mcp")
        self.mcp_log_handler = log_handler or self._default_mcp_log_handler

        if isinstance(mcp_config, FastMCPServer):
            self.client_logger.info(
                f"Using FastMCP server transport: {mcp_config.name}",
                extra={
                    'transport_type': 'fastmcp_server',
                    'server_name': mcp_config.name
                }
            )
            self.mcp_server_config = mcp_config
        else:
            self.client_logger.info(
                f"Using MCP config as transport: {mcp_config}",
                extra={
                    'transport_type': 'mcp_config',
                    'mcp_config': mcp_config
                }
            )
            self.mcp_server_config = copy.deepcopy(mcp_config)
            if mcp_server_query_params:
                self.client_logger.info(f"Adding MCP server query params to MCP config")
                self.mcp_server_config = self._add_mcp_server_query_params(mcp_server_query_params)

        # Store kwargs for client recreation
        self.client_init_kwargs = kwargs
        
        # Pass MCP log handler to underlying MCP client using factory method
        self.client = self._create_mcp_client(self.mcp_server_config)
        
        # Store on_tool_error and tool_result_handler
        self.on_tool_error = on_tool_error
        self.tool_result_handler = tool_result_handler or self._default_tool_result_handler


    def _create_mcp_client(self, transport) -> MCPClient:
        """Factory method for creating MCPClient instances with consistent config"""
        return MCPClient(
            transport=transport,
            log_handler=self.mcp_log_handler,
            **self.client_init_kwargs
        )


    async def _default_mcp_log_handler(self, message: LogMessage):
        """Handle MCP server logs using dedicated MCP logger"""
        LOGGING_LEVEL_MAP = logging.getLevelNamesMapping()
        msg = message.data.get('msg')
        extra = message.data.get('extra')
        extra = extra or {}
        extra.update({
            'source': 'mcp_server'
        })

        level = LOGGING_LEVEL_MAP.get(message.level.upper(), logging.INFO)
        self.mcp_logger.log(level, f"[MCP] {msg}", extra=extra)


    async def get_tools(self, format: Literal["mcp", "openai"] = "mcp") -> ListToolsResult:
        self.client_logger.debug(f"Getting MCP tools in {format} format")
        async with self.client:
            tools = await self.client.list_tools()
        if format == "mcp":
            self.client_logger.debug(f"MCP tools: {tools}")
            return tools
        elif format == "openai":
            self.client_logger.debug(f"Converting MCP tools to OpenAI format")
            openai_tools = [transform_mcp_tool_to_openai_tool(tool) for tool in tools]
            self.client_logger.debug(f"OpenAI tools: {openai_tools}")
            return openai_tools
        else:
            raise ValueError(f"Invalid tool list format. Expected 'mcp' or 'openai', got {format}")


    async def get_tool_input_format(self, tool_name: str):
        tools = await self.get_tools(format="mcp")
        for tool in tools:
            if tool.name == tool_name:
                return tool.inputSchema
        raise ValueError(f"Tool {tool_name} not found")

    
    def transform_tool_call_request(self, tool_call: ChatCompletionMessageToolCall) -> MCPCallToolRequestParams:
        self.client_logger.debug(f"Transforming tool call request to MCP format: {tool_call}")
        transformed_tool_call = transform_openai_tool_call_request_to_mcp_tool_call_request(openai_tool=tool_call.model_dump())
        transformed_tool_call.id = tool_call.id
        self.client_logger.debug(f"Transformed tool call request to MCP format: {transformed_tool_call}")
        return transformed_tool_call
            

    async def call_tool(self, tool_call: MCPCallToolRequestParams) -> PocketAgentToolResult:
        tool_call_id = tool_call.id
        tool_call_name = tool_call.name
        tool_call_arguments = tool_call.arguments
        tool_call_meta = tool_call.meta

        try:
            async with self.client:
                self.client_logger.debug(f"Calling tool: {tool_call_name} with arguments: {tool_call_arguments} and meta: {tool_call_meta}")
                tool_result = await self.client.call_tool(tool_call_name, tool_call_arguments, meta=tool_call_meta)
        except ToolError as e:
            # handle tool error
            if self.on_tool_error:
                on_tool_error_result = await self.on_tool_error(tool_call, e)
                if type(on_tool_error_result) == str:
                    tool_result_content = [{
                        "type": "text",
                        "text": on_tool_error_result
                    }]
                    return PocketAgentToolResult(
                        tool_call_id=tool_call_id,
                        tool_call_name=tool_call_name,
                        tool_result_content=tool_result_content
                    )

                # False means the tool error handler did not handle the error
                elif on_tool_error_result == False:
                    self.client_logger.error(
                        f"Tool error handler returned False, which means the tool call should be skipped: '{tool_call_name}'",
                        extra={
                            'tool_name': tool_call_name,
                            'tool_arguments': str(tool_call.arguments)[:200],
                            'error_type': type(e).__name__,
                            'error_message': str(e)
                        }
                    )
                    raise e

                # If we get here, the tool error handler returned an invalid type
                else:
                    error_message = f"Tool error handler returned invalid type: expected str|False, got {type(on_tool_error_result)}"
                    self.client_logger.error(
                        error_message,
                        extra={
                            'tool_name': tool_call_name,
                            'handler_result_type': type(on_tool_error_result).__name__,
                            'handler_result': str(on_tool_error_result)[:100]
                        }
                    )
                    raise ValueError(error_message)
            
            # If we get here, there is no tool error handler
            else:
                self.client_logger.error(
                    f"No tool error handler is set",
                    extra={
                        'tool_name': tool_call_name,
                        'tool_arguments': str(tool_call.arguments)[:200],
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    }
                )
                raise e
        else:
            return self.tool_result_handler(tool_call, tool_result)

    
    def _default_tool_result_handler(self, tool_call: ChatCompletionMessageToolCall, tool_result: FastMCPCallToolResult) -> PocketAgentToolResult:
        """
        The function transforms the fastmcp tool result to a tool result that can be used by the agent.
        The default implementation just extracts text and image content and transforms them into a format that can directly be used as a message
        """
        tool_result_content = []
        tool_result_raw_content = []
        for content in tool_result.content:
            tool_result_raw_content.append(content.model_dump())
            if content.type == "text":
                tool_result_content.append({
                    "type": "text",
                    "text": content.text
                })
            elif content.type == "image":
                tool_result_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content.mimeType};base64,{content.data}"
                    }
                })
        return PocketAgentToolResult(
            tool_call_id=tool_call.id,
            tool_call_name=tool_call.name,
            tool_result_content=tool_result_content,
            _extra={
                "tool_result_raw_content": tool_result_raw_content
            }
        )

    
    def mount_server(self, server: FastMCPServer):
        proxy_server = FastMCPServer.as_proxy(
            self.client
        )
        proxy_server.mount(server)
        self.client = self._create_mcp_client(proxy_server)




        

```

---

## 🔧 utils/console_formatter.py

**File Path:** `utils/console_formatter.py`  
**Stats:** 94 lines, 4,048 characters

*Console output formatting for agent events and interactions*

```python
import json
from typing import Any, Dict
from ..agent import AgentEvent

class ConsoleFormatter:
    """Handles formatting of agent events for console output"""
    
    def format_event(self, event: AgentEvent) -> None:
        if event.event_type == "new_message":
            role = event.data["role"]
            name = event.meta.get("agent_name", None)
            content = event.data.get("content", None)
            tool_call_id = event.data.get("tool_call_id", None)
            tool_calls = event.data.get("tool_calls", None)
            
            # Create a clean separator and header
            print("\n" + "=" * 60)
            if role == "tool":
                print(f"🔧 TOOL RESULT: (ID: {tool_call_id})")
            elif role == "user":
                print(f"👤 USER MESSAGE:")
            elif role == "assistant":
                print(f"🤖 ASSISTANT {name} MESSAGE:")
            else:
                print(f"🤖 {role.upper()} MESSAGE")
            print("=" * 60)
            
            # Handle content formatting
            if content:
                if isinstance(content, str):
                    # Simple string content
                    self._print_formatted_text(content)
                else:
                    # List content (multimodal)
                    for item in content:
                        if item.get("type") == "text":
                            self._print_formatted_text(item.get('text', ''))
                        elif item.get("type") == "image_url":
                            print("   🖼️  [Image attached]")
                        else:
                            print(f"   📎 [{item.get('type', 'unknown').title()} content]")
            
            # Handle tool calls formatting
            if tool_calls:
                print("\n🔧 Tool Calls:")
                for i, tool_call in enumerate(tool_calls, 1):
                    function_info = tool_call.get('function', {})
                    function_name = function_info.get('name', 'unknown')
                    function_args = function_info.get('arguments', '{}')
                    tool_call_id = tool_call.get('id', None)
                    print(f"   [{i}] {function_name} (ID: {tool_call_id})")
                    try:
                        args_dict = json.loads(function_args) if isinstance(function_args, str) else function_args
                        if args_dict:
                            for key, value in args_dict.items():
                                # Truncate long values for readability
                                if isinstance(value, str) and len(value) > 100:
                                    display_value = value[:97] + "..."
                                else:
                                    display_value = value
                                print(f"       {key}: {display_value}")
                        else:
                            print("       (no arguments)")
                    except (json.JSONDecodeError, TypeError):
                        print(f"       arguments: {function_args}")
            
            print("=" * 60 + "\n")
            
        else:
            # Handle other event types
            print(f"\n📡 Event: {event.event_type}")
            if event.data:
                print(f"   Data: {json.dumps(event.data, indent=2)}")
            print()
    
    def _print_formatted_text(self, text: str) -> None:
        """Helper method to format text content with proper indentation"""
        if not text:
            return
        # try to json load the text
        try:
            json_text = json.loads(text)
            text = json.dumps(json_text, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass
        # Split into lines and add indentation
        lines = text.strip().split('\n')
        for line in lines:
            if line.strip():  # Only print non-empty lines
                print(f"   {line}")
            else:
                print()  # Preserve empty lines for spacing
    

```

---

## 🔧 utils/logger.py

**File Path:** `utils/logger.py`  
**Stats:** 152 lines, 5,443 characters

*Professional logging configuration with file rotation and level management*

```python
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Union

class PocketAgentLogger:
    """Professional logging configuration for PocketAgent framework"""
    
    def __init__(self, 
                 name: str = "pocket_agent",
                 level: Union[str, int] = logging.INFO,
                 log_dir: Optional[Path] = None,
                 console: bool = True,
                 file_logging: bool = True,
                 max_bytes: int = 10*1024*1024,  # 10MB
                 backup_count: int = 5):
        
        self.name = name
        self.level = self._parse_level(level)
        self.log_dir = log_dir or Path.cwd()
        self.console = console
        self.file_logging = file_logging
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create the main logger (don't modify root!)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _parse_level(self, level: Union[str, int]) -> int:
        if isinstance(level, str):
            return getattr(logging, level.upper(), logging.INFO)
        return level
    
    def _setup_handlers(self):
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.level)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.file_logging:
            log_file = self.log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.level)
            self.logger.addHandler(file_handler)
    
    def get_logger(self, component: str = "") -> logging.Logger:
        """Get a component-specific logger"""
        if component:
            logger_name = f"{self.name}.{component}"
        else:
            logger_name = self.name
        
        return logging.getLogger(logger_name)
    
    def configure_third_party_loggers(self):
        """Configure third-party library loggers without interfering"""
        third_party_configs = {
            'fastmcp': logging.WARNING,
            'mcp': logging.WARNING, 
            'litellm': logging.WARNING,
            'urllib3': logging.WARNING,  # Often noisy
            'httpx': logging.WARNING,
        }
        
        for logger_name, level in third_party_configs.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)

def _detect_environment():
    """Detect if we're in development vs production"""
    # Common development indicators
    dev_indicators = [
        os.getenv("DEBUG"),
        os.getenv("DEVELOPMENT"), 
        os.path.exists(".git"),  # In a git repo
        "pytest" in sys.modules,  # Running tests
        "jupyter" in sys.modules,  # In Jupyter
    ]
    return any(dev_indicators)

def configure_logger(log_level: str = "INFO", console_level: str = "WARNING", name: str = "pocket_agent"):
    """
    Configure hierarchical logging for PocketAgent framework
    
    Args:
        log_level: Overall logging level (affects file output)
        console_level: Console output level (default: WARNING, so only warnings/errors show)
    """
    log_level = os.getenv("POCKET_AGENT_LOG_LEVEL", log_level)
    console_level = os.getenv("POCKET_AGENT_CONSOLE_LEVEL", console_level)
    
    log_level = log_level.upper()
    console_level = console_level.upper()
    
    log_level_int = logging.getLevelNamesMapping().get(log_level, logging.INFO)
    console_level_int = logging.getLevelNamesMapping().get(console_level, logging.WARNING)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get or create the main logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level_int)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Add console handler - WARNING+ by default (errors/warnings visible immediately)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level_int)  # Only warnings/errors by default
    logger.addHandler(console_handler)
    
    # Add file handler - full detail based on log_level
    file_handler = logging.handlers.RotatingFileHandler(
        'pocket-agent.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level_int)  # Full detail in file
    logger.addHandler(file_handler)
    
    # Configure third-party loggers (keep them quiet)
    for lib_name in ['fastmcp', 'mcp', 'litellm', 'urllib3', 'httpx']:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.WARNING)
    
    return logger
```

---

## 📊 Complete Dump Summary

### 📋 **Content Overview:**
- **📖 Documentation Files:** 9 files ✓
- **🔧 Source Code Files:** 5 files ✓
- **📏 Total Content:** 1,752 lines (69,112 characters)

### 🏗️ **Framework Architecture:**
- **🤖 Core Agent (`agent.py`)** - Main orchestration and lifecycle management
- **🔌 Client Layer (`client.py`)** - MCP server communication and tool execution  
- **🛠️ Utilities** - Logging and console formatting for developer experience
- **📦 Public API (`__init__.py`)** - Clean, simple imports for end users
- **⚙️ CLI Tools** - Documentation and source code dumping utilities

### 🚀 **Key Features:**
- Multi-agent orchestration and sub-agent support
- Async/await throughout for performance
- Comprehensive error handling and logging
- Hook system for extensibility
- Clean separation of concerns
- Complete documentation and examples

---

*This complete dump was generated by `pocket-dump` - documentation and source code dumping utility.*

**Need another dump?** Run: `pocket-dump <filename>`  
**Just source?** Run: `pocket-dump --source-only <filename>`  
**Just docs?** Run: `pocket-dump --docs-only <filename>`

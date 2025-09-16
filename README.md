<div align="center">

# Pocket-Agent

<img src="./assets/pocket-agent.png" alt="Pocket Agent" width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

<p><em>A lightweight, extensible framework for building LLM agents with Model Context Protocol (MCP) support</em></p>

[![PyPI - Version](https://img.shields.io/pypi/v/fastmcp.svg)](https://pypi.org/project/pocket-agent/)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

---

## Table of Contents

- [Why Pocket Agent?](#why-pocket-agent)
- [Design Principles](#design-principles)
- [Cookbook](#-cookbook)
- [Installation](#installation)
- [Creating Your First Pocket-Agent (Quick Start)](#creating-your-first-pocket-agent-quick-start)
- [Building Pocket-Agents with Agents](#building-pocket-agents-with-agents)
- [Core Concepts](#core-concepts)
  - [PocketAgent Base Class](#️-pocketagent-base-class)
  - [The Step Method](#-the-step-method)
  - [Message Management](#-message-management)
  - [Hook System](#-hook-system)
  - [Event System](#-event-system)
  - [Multi-Model Support](#-multi-model-support)
  - [LiteLLM Router Integration](#-litellm-router-integration)
  - [PocketAgentClient](#pocketagentclient)
- [Testing](#testing)
- [Feature Roadmap](#feature-roadmap)

---

## Why Pocket Agent?

Most agent frameworks are severely over-bloated. The reason for this is that they are trying to support too many things at once and make every possible agent implementation "simple". This only works until it doesn't and you are stuck having to understand the enormous code base to implement what should be a very simple feature.

Pocket Agent takes the opposite approach by handling only the basic functions of an LLM agent and working with the MCP protocol. That way you don't give up any flexibility when building your agent but a lot of the lower level implementation details are taken care of.


## Design Principles

### 🚀 **Lightweight & Simple**
- Minimal dependencies - just `fastmcp` and `litellm`
- Clean abstractions that separate agent logic from MCP client details  
- < 500 lines of code

### 🎯 **Developer-Friendly**
- Abstract base class design for easy extension
- Clear separation of concerns between agents and clients
- Built-in logging and event system

### 🌐 **Multi-Model Support**
- Works with any endpoint supported by LiteLLM without requiring code changes
- Easy model switching and configuration

### 💡 **Extensible**
- Use any custom logging implementation
- Easily integrate custom frontends using the built-in event system
- Easily create fully custom agent implementations

## 🧑‍🍳 [Cookbook](https://github.com/DIR-LAB/pocket-agent/tree/main/cookbook)
#### Refer to the [Cookbook](https://github.com/DIR-LAB/pocket-agent/tree/main/cookbook) to find example implementations and try out PocketAgent without any implementation overhead


## Installation

Install with uv (Recommended):
```bash
uv add pocket-agent
```

Install with pip:
```bash
pip install pocket-agent
```

## Creating Your First Pocket-Agent (Quick Start)

#### To build a Pocket-Agent, all you need to implement is the agent's `run` method:

```python
class SimpleAgent(PocketAgent):
    async def run(self):
        """Simple conversation loop"""

        while True:
            # Accept user message
            user_input = input("Your input: ")
            if user_input.lower() == 'quit':
                break
                
            # Add user message
            await self.add_user_message(user_input)
            
             # Generates response and executes any tool calls
            step_result = await self.step()
            while step_result["llm_message"].tool_calls is not None:
                step_result = await self.step()
    
        return {"status": "completed"}
```

#### To run the agent, you only need to pass your [JSON MCP config](https://gofastmcp.com/integrations/mcp-json-configuration) and your agent configuration:

```python
mcp_config = {
    "mcpServers": {
        "weather": {
            "transport": "stdio",
            "command": "python",
            "args": ["server.py"],
            "cwd": os.path.dirname(os.path.abspath(__file__))
        }
    }
}
# Configure agent  
config = AgentConfig(
    llm_model="gpt-5-nano",
    system_prompt="You are a helpful assistant who answers user questions and uses provided tools when applicable"
)
# Create and run agent
agent = SimpleAgent(
    agent_config=config,
    mcp_config=mcp_config
)

await agent.run()
```

## Building Pocket-Agents with Agents
If you are using an agent (i.e. cursor, claude) to your PocketAgent, you can provide the agent with the [llm.md](./llm.md) as useful documentation.


## Core Concepts

### 🏗️ **PocketAgent Base Class**

The `PocketAgent` is an abstract base class that provides the foundation for building custom agents. You inherit from this class and implement the `run()` method to define your agent's behavior.

```python
from pocket_agent import PocketAgent, AgentConfig

class MyAgent(PocketAgent):
    async def run(self):
        # Your agent logic here
        return {"status": "completed"}
```

**PocketAgent Parameters:**
```python
agent = PocketAgent(
    agent_config,   # Required: Instance of the AgentConfig class
    mcp_config,     # Required: JSON MCP server configuration to pass tools to the agent
    router,         # Optional: A litellm router to manage llm rate limits
    logger,         # Optional: A logger instance to capture logs
    hooks,          # Optional: Instance of AgentHooks to optionally define custom behavior at common junction points
    **client_kwargs # Optional: additional kwargs passed to the PocketAgentClient

)
```

**AgentConfig Parameters:**

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

### 🔄 **The Step Method**

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

**Step Result Structure:**
```python
{
    "llm_message": LitellmMessage,           # The LLM response
    "tool_execution_results": [ToolResult]   # Results from tool calls (if any)
}
```

### 💬 **Message Management**

Pocket Agent automatically adds llm generated messages and tool result messages in the `step()` function.
Input provided by a user can easily be managed using `add_user_message()` and should be done before calling the `step()` method:

```python
class Agent(PocketAgent)
    async def run(self):
        # Add user messages (with optional images)
        await agent.add_user_message("Hello!", image_base64s=["base64_image_data"])
        await self.step()

# Clear all messages except the system promp `reset_messages` function
agent.reset_messages()
```

### 🪝 **Hook System**

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

### 📡 **Event System**

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

### 🔧 **Multi-Model Support**

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

### 🚏 **LiteLLM Router Integration**
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

### PocketAgentClient

Each PocketAgent instance creates initializes a PocketAgentClient which acts as a wrapper for the [FastMCP Client](https://gofastmcp.com/clients/client) to implement the standard mcp protocol features and some additional features.

**Custom Query Params**

- Sending metadata such as a custom id to MCP servers is not handled well by the protocol (until [this](https://github.com/modelcontextprotocol/python-sdk/pull/1231) is merged). For now a workaround is to send metadata via query params to mcp servers using an http transport.

    ```python
    agent = PocketAgent(
        mcp_server_query_params = {
            "context_id": "1111"    # context_id will automatically be added to the server endpoint when sending a request
        }
    )
    ```

    *Note: Query params must use custom MCP middleware to be interpreted by servers*

**on_tool_error (hook)**

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

**tool_result_handler (hook)**

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

**Server-initiated Events**
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


You can also provide a custom LiteLLM Router for advanced model routing and fallback logic.


## Testing

Pocket Agent includes a comprehensive test suite covering all core functionality. The tests are designed to be fast and reliable using in-memory FastMCP servers and mocked LLM responses.

### Running Tests

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

## Feature Roadmap

### Core Features
| Feature | Status | Priority | Description |
|---------|--------|----------|-------------|
| **Agent Abstraction** | ✅ Implemented | - | Basic agent abstraction with PocketAgent base class |
| **MCP Protocol Support** | ✅ Implemented | - | Full integration with Model Context Protocol via fastmcp |
| **Multi-Model Support** | ✅ Implemented | - | Support for any LiteLLM compatible model/endpoint |
| **Tool Execution** | ✅ Implemented | - | Automatic parallel tool calling and results handling |
| **Hook System** | ✅ Implemented | - | Allow configurable hooks to inject functionality during agent execution |
| **Logging Integration** | ✅ Implemented | - | Built-in logging with custom logger support |
| **Streaming Responses** | 📋 Planned | Medium | Real-time response streaming support |
| **Define Defaults for standard MCP Client handlers | 📋 Planned | Medium | Standard MCP client methods (i.e. sampling, progress, etc) may benefit from default implementations if custom behavior is not often needed |
| **Multi-Agent Integration** | 📋 Planned | High | Allow a PocketAgent to accept other PocketAgents as Sub Agents and automatically set up Sub Agents as tools for the Agent to use |
| **Resources Integration** | 📋 Planned | Medium | Automatically set up mcp read_resource functionality as a tool |

### Modality support
| Modality | Status | Priority | Description |
|---------|--------|----------|-------------|
| **Text** | ✅ Implemented | - | Multi-modal input support for vision models |
| **Images** | ✅ Implemented | - | Multi-modal input support for VLMs with option to enable/disable |
| **Audio** | 📋 Planned | Low | Multi-modal input support for LLMs which allow audio inputs |


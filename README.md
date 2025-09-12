<div align="center">

# Pocket-Agent

<img src="./assets/pocket-agent.png" alt="Pocket Agent" width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

<p><em>A lightweight, extensible framework for building LLM agents with Model Context Protocol (MCP) support</em></p>

![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

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

## [Cookbook](./cookbook)
#### Refer to the [Cookbook](./cookbook) to find example implementations and try out PocketAgent without any implementation overhead


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
            self.add_user_message(user_input)
            
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

**PocketAgent Input Parameters:**
```python
agent = (
    agent_config,   # Required: Instance of the AgentConfig class
    mcp_config,     # Required: JSON MCP server configuration to pass tools to the agent
    router,         # Optional: A litellm router to manage llm rate limits
    logger,         # Optional: A logger instance to capture logs
    hooks,          # Optional: Instance of AgentHooks to optionally define custom behavior at common junction points

)
```

### ⚙️ **AgentConfig**

Configuration object that defines your agent's setup and behavior:

```python
config = AgentConfig(
    llm_model="gpt-4",                    # Required: LLM model to use
    system_prompt="You are helpful",      # Optional: System prompt for the agent
    agent_id="my-agent-123",              # Optional: Custom agent ID
    context_id="conversation-456",        # Optional: Custom context ID
    allow_images=True,                    # Optional: Enable image input support (default: False)
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

# Handle tool calls (continue until no more tool calls)
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
    def run(self):
        # Add user messages (with optional images)
        agent.add_user_message("Hello!", image_base64s=["base64_image_data"])
        self.step()

# Clear all messages except the system promp `reset_messages` function
agent.reset_messages()
```

**Message Format:** Standard OpenAI message format with role, content, and optional tool metadata.

### 🛠️ **MCP Integration via PocketAgentClient**

The `PocketAgentClient` handles all MCP server communication:
- **Tool Discovery**: Automatically fetches available tools from MCP servers
- **Tool Execution**: Transforms OpenAI tool calls to MCP format and handles execution
- **Parallel Execution**: Executes multiple tool calls simultaneously
- **Error Handling**: Provides hooks for custom error handling

```python
# MCP configuration format
mcp_config = {
    "mcpServers": {
        "weather": {
            "transport": "stdio",
            "command": "python",
            "args": ["weather_server.py"]
        },
        "web": {
            "transport": "sse",
            "url": "http://localhost:3001/sse"
        }
    }
}
```

### 🪝 **Hook System for Extensibility**

Customize agent behavior at key execution points:

```python
class CustomHooks(AgentHooks):
    def pre_step(self, context):
        # executed before the llm response is generated in the step() method
        print("About to execute step")
    
    def post_step(self, context):
        # executed after all tool results (if any) are retrieved; This runs even if tool calling results in an error
        print("Step completed")
    
    def pre_tool_call(self, context, tool_call):
        # executed right before a tool is run
        print(f"Calling tool: {tool_call.name}")
        # Return modified tool_call or None
    
    def post_tool_call(self, context, tool_call, result):
        # executed right after a tool call result is retrieved from the PocketAgentClient
        print(f"Tool {tool_call.name} completed")
        return result  # Return modified result
    
    def on_llm_response(self, context, response):
        # executed right after a response message has been generated by the llm
        print("Got LLM response")
    
    def on_event(self, event: AgentEvent):
        # Default behavior is to run when any new message is added
        print(f"Event: {event.event_type}")

# Use custom hooks
agent = MyAgent(
    agent_config=config,
    mcp_config=mcp_config,
    hooks=CustomHooks()
)
```

### 📡 **Event System**

Built-in event system for monitoring and integration:

```python
@dataclass
class AgentEvent:
    event_type: str  # e.g., "new_message", "tool_call_start"
    data: dict       # Event-specific data
```

Events are automatically emitted for:
- New messages added to conversation
- Tool calls and completions
- Agent lifecycle events

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

You can also provide a custom LiteLLM Router for advanced model routing and fallback logic.

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


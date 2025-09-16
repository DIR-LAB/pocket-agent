# Guide to Pocket Agent Framework

This guide provides comprehensive information for AI assistants working with the Pocket Agent framework - a lightweight, extensible framework for building LLM agents with Model Context Protocol (MCP) support.

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Core Architecture](#core-architecture)
3. [Key Classes and Components](#key-classes-and-components)
4. [Implementation Patterns](#implementation-patterns)
5. [Best Practices](#best-practices)
6. [Common Tasks and Solutions](#common-tasks-and-solutions)
7. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
8. [Advanced Features](#advanced-features)

## Framework Overview

### Design Philosophy

Pocket Agent follows a **minimal but extensible** approach:
- **Lightweight**: < 500 lines of core code, minimal dependencies (`fastmcp` + `litellm`)
- **Flexible**: Abstract base class design allows complete customization of agent behavior
- **Multi-modal**: Built-in support for text and images, with audio support planned
- **Protocol-native**: Deep integration with Model Context Protocol (MCP) for tool usage

### Key Benefits
- Clean separation between agent logic and MCP client details
- Built-in event system for frontend integration
- Automatic parallel tool execution
- Comprehensive hook system for customization
- Multi-model support via LiteLLM

## Core Architecture

┌─────────────────────────────────────────────────────────┐
│                    PocketAgent                          │
├─────────────────────────────────────────────────────────┤
│  • run() method (abstract - you implement)             │
│  • step() method (handles LLM + tools)                 │
│  • Message management                                   │
│  • Hook system integration                              │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 PocketAgentClient                       │
├─────────────────────────────────────────────────────────┤
│  • MCP tool execution                                   │
│  • Tool result transformation                           │
│  • Error handling                                       │
│  • FastMCP Client wrapper                               │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   FastMCP Client                        │
├─────────────────────────────────────────────────────────┤
│  • MCP protocol implementation                          │
│  • Server communication                                 │
│  • Transport management                                 │
└─────────────────────────────────────────────────────────┘

## Key Classes and Components

### 1. PocketAgent (Abstract Base Class)

```python
class MyAgent(PocketAgent):
    async def run(self):
        """Your main agent logic goes here"""
        return {"status": "completed"}
```

**Key Methods:**
- `run()` - **Abstract method you must implement** (returns dict)
- `step()` - Core execution unit (LLM response + tool calls, returns StepResult)
- `add_user_message(text, image_base64s=None)` - Add user input
- `reset_messages()` - Clear conversation history

### 2. AgentConfig

Configuration object for agent setup:

```python
config = AgentConfig(
    llm_model="gpt-4",                    # Required: Model identifier
    system_prompt="You are helpful...",   # Optional: System prompt
    agent_id="custom-id",                 # Optional: Custom identifier
    allow_images=True,                    # Optional: Enable image support
    messages=[],                          # Optional: Initial conversation
    completion_kwargs={                   # Optional: LLM parameters
        "tool_choice": "auto",
        "temperature": 0.7
    }
)
```

### 3. StepResult

Return value from `step()` method:

```python
@dataclass
class StepResult:
    llm_message: LitellmMessage                                 # LLM response
    tool_execution_results: Optional[list[ToolResult]] = None   # Tool results
```

### 4. AgentHooks

Hook system for customizing behavior:

```python
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from pocket_agent import ToolResult, AgentHooks, HookContext, AgentEvent
from litellm.types.utils import ModelResponse as LitellmModelResponse

class CustomHooks(AgentHooks):
    async def pre_step(self, context: HookContext) -> None:
        # Called before LLM response generation
        pass
    
    async def post_step(self, context: HookContext) -> None:
        # Called after step completion
        pass
    
    async def pre_tool_call(self, context: HookContext, tool_call: MCPCallToolRequestParams) -> Optional[MCPCallToolRequestParams]:
        # Called before each tool execution
        return None  # Return modified tool_call or None to keep original
    
    async def post_tool_call(self, context: HookContext, tool_call: MCPCallToolRequestParams, result: ToolResult) -> Optional[ToolResult]:
        # Called after each tool execution
        return result  # Return modified result
    
    async def on_llm_response(self, context: HookContext, response: LitellmModelResponse) -> None:
        # Called after LLM generates response
        pass
    
    async def on_event(self, event: AgentEvent) -> None:
        # Called for agent events (new messages, etc.)
        pass

    # tool result handler will replace the default tool result handler in PocketAgentClient if implemented
    # async def on_tool_result(self, context: HookContext, tool_call: ChatCompletionMessageToolCall, tool_result: FastMCPCallToolResult) -> ToolResult:
    #     pass
    
    async def on_tool_error(self, context: HookContext, tool_call: MCPCallToolRequestParams, error: Exception) -> Union[str, False]:
        if "unexpected_keyword_argument" in str(error):
            tool_call_name = tool_call.name
            tool_format = await context.agent.mcp_client.get_tool_input_format(tool_call_name)
            return "You supplied an unexpected keyword argument to the tool. \
                Try again with the correct arguments as specified in expected format: \n" + json.dumps(tool_format)
        return False
```

## Implementation Patterns

### 1. Basic Conversational Agent

```python
class ChatAgent(PocketAgent):
    async def run(self):
        """Simple conversation loop"""
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'quit':
                break
                
            # Add user message
            await self.add_user_message(user_input)
            
            # Process with tools
            step_result = await self.step()
            while step_result.llm_message.tool_calls:
                step_result = await self.step()
                
            # Display response
            print(f"Agent: {step_result.llm_message.content}")
        
        return {"status": "completed"}
```

### 2. Task-Oriented Agent

```python
class TaskAgent(PocketAgent):
    async def run(self):
        """Single task execution"""
        # Process initial instruction
        await self.add_user_message(self.initial_task)
        
        # Execute until no more tool calls
        step_result = await self.step()
        while step_result.llm_message.tool_calls:
            step_result = await self.step()
        
        # Return final result
        return {
            "status": "completed",
            "result": step_result.llm_message.content,
            "tool_calls_made": len(step_result.tool_execution_results or [])
        }
```

### 3. Multi-Turn Agent with Context

```python
class ContextualAgent(PocketAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_context = {}
    
    async def run(self):
        """Multi-turn conversation with context tracking"""
        for turn in self.conversation_turns:
            await self.add_user_message(turn["input"])
            
            # Process with context
            step_result = await self.step()
            while step_result.llm_message.tool_calls:
                step_result = await self.step()
            
            # Update context
            self.conversation_context[turn["id"]] = {
                "response": step_result.llm_message.content,
                "tools_used": [r.tool_call_name for r in (step_result.tool_execution_results or [])]
            }
        
        return {"context": self.conversation_context}
```

## Best Practices

### 1. Agent Design

**Do:**
- Implement error handling in your `run()` method
- Use meaningful system prompts that guide tool usage
- Structure your agent logic with clear phases
- Use the hook system for cross-cutting concerns
- Log important decisions and state changes

**Don't:**
- Forget to handle the case where `step_result.tool_execution_results` is None
- Ignore tool call errors without proper handling
- Mix agent logic with UI/presentation concerns
- Forget to reset messages when needed

### 2. Tool Integration

**MCP Configuration Best Practices:**

```python
mcp_config = {
    "mcpServers": {
        "server_name": {
            "transport": "stdio",           # or "http"
            "command": "python",            # for stdio
            "args": ["server.py"],          # server arguments
            "cwd": "/path/to/server",       # working directory
            # For HTTP:
            # "url": "http://localhost:8080"
        }
    }
}
```

**Tool Call Pattern:**

```python
# Always handle the tool call loop properly
step_result = await self.step()
while step_result.llm_message.tool_calls is not None:
    self.logger.info(f"Executing {len(step_result.llm_message.tool_calls)} tool calls")
    step_result = await self.step()

# Final response is in step_result.llm_message.content
```

### 3. Error Handling

```python
class RobustAgent(PocketAgent):
    async def run(self):
        try:
            await self.add_user_message(self.task)
            
            step_result = await self.step()
            max_iterations = 10
            iteration = 0
            
            while step_result.llm_message.tool_calls and iteration < max_iterations:
                iteration += 1
                step_result = await self.step()
            
            if iteration >= max_iterations:
                return {"status": "max_iterations_reached", "partial_result": step_result.llm_message.content}
                
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            return {"status": "error", "error": str(e)}
        
        return {"status": "completed", "result": step_result.llm_message.content}
```

### 4. Hook Usage

```python
from pocket_agent import AgentHooks, HookContext, AgentEvent
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams

class MonitoringHooks(AgentHooks):
    def __init__(self):
        self.tool_usage = {}
        self.step_count = 0
    
    async def pre_step(self, context: HookContext):
        self.step_count += 1
        context.metadata["step_number"] = self.step_count
    
    async def pre_tool_call(self, context: HookContext, tool_call):
        tool_name = tool_call.name
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
        context.agent.logger.info(f"Using tool {tool_name} (usage count: {self.tool_usage[tool_name]})")
        return tool_call
    
    async def on_event(self, event: AgentEvent):
        if event.event_type == "new_message":
            print(f"New message: {event.data.get('role', 'unknown')}")
```

## Common Tasks and Solutions

### 1. Image Processing Agent

```python
class ImageAnalysisAgent(PocketAgent):
    async def run(self):
        """Process images with vision capabilities"""
        for image_path in self.image_paths:
            # Load and encode image
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode()
            
            await self.add_user_message(
                "Analyze this image and describe what you see",
                image_base64s=[image_base64]
            )
            
            step_result = await self.step()
            while step_result.llm_message.tool_calls:
                step_result = await self.step()
            
            # Store analysis result
            self.results[image_path] = step_result.llm_message.content
        
        return {"analyses": self.results}

# Configure with vision model and enable images
config = AgentConfig(
    llm_model="gpt-4o",  # or other vision-capable model
    allow_images=True,
    system_prompt="You are an expert image analyst..."
)
```

### 2. Batch Processing Agent

```python
class BatchProcessor(PocketAgent):
    async def run(self):
        """Process multiple tasks in sequence"""
        results = []
        
        for task in self.tasks:
            # Reset for each task
            self.reset_messages()
            
            await self.add_user_message(task["prompt"])
            
            step_result = await self.step()
            while step_result.llm_message.tool_calls:
                step_result = await self.step()
            
            results.append({
                "task_id": task["id"],
                "result": step_result.llm_message.content,
                "tools_used": [r.tool_call_name for r in (step_result.tool_execution_results or [])]
            })
        
        return {"batch_results": results}
```

### 3. Agent with Custom Tool Error Handling

```python
from pocket_agent import AgentHooks, HookContext
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from typing import Union

class CustomErrorHooks(AgentHooks):
    async def on_tool_error(self, context: HookContext, tool_call: MCPCallToolRequestParams, error: Exception) -> Union[str, False]:
        # Log the error
        context.agent.logger.error(f"Tool {tool_call.name} failed: {error}")
        
        # Provide helpful error message to LLM
        if "permission" in str(error).lower():
            return f"Permission denied for tool {tool_call.name}. Please try a different approach."
        elif "not found" in str(error).lower():
            return f"Resource not found for tool {tool_call.name}. Please check your parameters."
        else:
            return f"Tool {tool_call.name} encountered an error. Please try with different parameters or use an alternative approach."
```

### 4. Streaming/Event-Driven Agent

```python
from pocket_agent import AgentHooks, HookContext, ToolResult
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams

class EventDrivenHooks(AgentHooks):
    def __init__(self, event_callback):
        self.event_callback = event_callback
    
    async def on_llm_response(self, context: HookContext, response):
        # Stream LLM responses
        await self.event_callback({
            "type": "llm_response",
            "content": response.choices[0].message.content,
            "has_tool_calls": bool(response.choices[0].message.tool_calls)
        })
    
    async def pre_tool_call(self, context: HookContext, tool_call):
        # Notify about tool usage
        await self.event_callback({
            "type": "tool_start",
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments
        })
        return tool_call
    
    async def post_tool_call(self, context: HookContext, tool_call, result):
        # Notify about tool completion
        await self.event_callback({
            "type": "tool_complete",
            "tool_name": tool_call.name,
            "result_preview": str(result.tool_result_content)[:100] + "..."
        })
        return result
```

## Debugging and Troubleshooting

### 1. Common Issues

**Issue: Tool calls not executing**
```python
# Check if tools are available
tools = await agent.mcp_client.get_tools(format="openai")
print(f"Available tools: {[tool.get('function', {}).get('name') for tool in tools]}")

# Verify MCP server configuration
print(f"MCP config: {agent.mcp_client.mcp_server_config}")
```

**Issue: Messages not formatted correctly**
```python
# Enable debug logging
import logging
logging.getLogger("pocket_agent").setLevel(logging.DEBUG)

# Check message history
print(f"Current messages: {agent.messages}")
print(f"Formatted messages: {agent._format_messages()}")
```

**Issue: Agent stuck in tool call loop**
```python
# Add iteration limits
max_iterations = 10
iteration = 0
step_result = await agent.step()

while step_result.llm_message.tool_calls and iteration < max_iterations:
    iteration += 1
    print(f"Tool call iteration {iteration}")
    step_result = await agent.step()

if iteration >= max_iterations:
    print("Warning: Max iterations reached")
```

### 2. Debugging Hooks

```python
from pocket_agent import AgentHooks, HookContext
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from litellm.types.utils import ModelResponse as LitellmModelResponse

class DebugHooks(AgentHooks):
    async def pre_step(self, context: HookContext):
        print(f"=== STEP START (Messages: {len(context.agent.messages)}) ===")
    
    async def post_step(self, context: HookContext):
        print(f"=== STEP END ===")
    
    async def pre_tool_call(self, context: HookContext, tool_call):
        print(f"🔧 Calling tool: {tool_call.name}")
        print(f"   Arguments: {tool_call.arguments}")
        return tool_call
    
    async def post_tool_call(self, context: HookContext, tool_call, result):
        print(f"✅ Tool {tool_call.name} completed")
        print(f"   Result length: {len(str(result.tool_result_content))}")
        return result
    
    async def on_llm_response(self, context: HookContext, response):
        message = response.choices[0].message
        print(f"🤖 LLM Response: {message.content[:100]}...")
        if message.tool_calls:
            print(f"   Tool calls: {len(message.tool_calls)}")
```

## Advanced Features

### 1. Custom LiteLLM Router

```python
from litellm import Router

# Rate limiting and fallback configuration
router_config = {
    "models": [
        {
            "model_name": "primary-model",
            "litellm_params": {
                "model": "gpt-4",
                "tpm": 40000,
                "rpm": 500
            }
        },
        {
            "model_name": "fallback-model",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "tpm": 80000,
                "rpm": 1000
            }
        }
    ],
    "routing_strategy": "least-busy"
}

router = Router(model_list=router_config["models"])

agent = MyAgent(
    agent_config=config,
    mcp_config=mcp_config,
    router=router
)
```

### 2. Custom MCP Server Query Parameters

```python
# Pass metadata to MCP servers via query params
agent = PocketAgent(
    agent_config=config,
    mcp_config=mcp_config,
    mcp_server_query_params={
        "user_id": "123",
        "session_id": "abc456",
        "context_id": "task_001"
    }
)
```

### 3. Custom Tool Result Processing

```python
class CustomResultHooks(AgentHooks):
    async def on_tool_result(self, context: HookContext, tool_call: ChatCompletionMessageToolCall, tool_result: FastMCPCallToolResult) -> ToolResult:
        # Custom processing for specific tools
        if tool_call.function.name == "database_query":
            # Parse structured database results
            parsed_data = self._parse_db_result(tool_result.content)
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_call_name=tool_call.function.name,
                tool_result_content=[{
                    "type": "text",
                    "text": f"Database returned {len(parsed_data)} records:\n{parsed_data}"
                }]
            )
        
        # Use default processing for other tools
        return context.agent.mcp_client._default_tool_result_handler(tool_call, tool_result)
```

### 4. Multi-Agent Integration (Future Feature)

```python
# This pattern will be supported in future versions
class CoordinatorAgent(PocketAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Sub-agents for specialized tasks
        self.research_agent = ResearchAgent(...)
        self.analysis_agent = AnalysisAgent(...)
        self.writing_agent = WritingAgent(...)
    
    async def run(self):
        """Coordinate multiple specialized agents"""
        # This is a conceptual example - not yet implemented
        research_result = await self.research_agent.run()
        analysis_result = await self.analysis_agent.run(research_result)
        final_result = await self.writing_agent.run(analysis_result)
        
        return final_result
```

## Framework Integration Tips

### 1. Web Framework Integration

```python
# FastAPI example
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

@app.post("/agent/run")
async def run_agent(task: dict, background_tasks: BackgroundTasks):
    config = AgentConfig(
        llm_model=task["model"],
        system_prompt=task["prompt"]
    )
    
    agent = TaskAgent(
        agent_config=config,
        mcp_config=load_mcp_config(),
        hooks=WebHooks()  # Custom hooks for web integration
    )
    
    # Run agent in background
    background_tasks.add_task(agent.run)
    
    return {"status": "started", "agent_id": agent.agent_id}
```

### 2. Database Integration

```python
class DatabaseIntegratedAgent(PocketAgent):
    def __init__(self, *args, db_session=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_session = db_session
    
    async def run(self):
        # Save initial state
        self.save_agent_state()
        
        try:
            result = await super().run()
            # Save final result
            self.save_agent_result(result)
            return result
        except Exception as e:
            # Save error state
            self.save_agent_error(e)
            raise
    
    def save_agent_state(self):
        # Database persistence logic
        pass
```

This guide covers the essential concepts and patterns for working effectively with the Pocket Agent framework. The framework's minimalist design makes it highly adaptable to various use cases while providing robust foundations for agent development.
```

The file has been created! This comprehensive guide covers all the key aspects of working with the pocket_agent framework, including:

- **Framework Overview**: Understanding the design philosophy and benefits
- **Core Architecture**: How the components work together  
- **Key Classes**: Detailed explanation of PocketAgent, AgentConfig, hooks, etc.
- **Implementation Patterns**: Common agent patterns with code examples
- **Best Practices**: Do's and don'ts for effective agent development
- **Common Tasks**: Practical examples for image processing, batch operations, etc.
- **Debugging**: Troubleshooting tips and debugging hooks
- **Advanced Features**: Router integration, custom result processing, etc.

This guide should serve as a comprehensive reference for anyone (especially AI assistants) working with the pocket_agent framework, providing both conceptual understanding and practical implementation guidance.

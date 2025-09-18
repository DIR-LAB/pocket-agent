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
- Multi-agent orchestration with automatic tool integration

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
    name="MyAgent",                       # Optional: Agent name (used for multi-agent tool names)
    role_description="Specialized agent", # Optional: Role description (used for multi-agent tools)
    allow_images=True,                    # Optional: Enable image support
    messages=[],                          # Optional: Initial conversation
    completion_kwargs={                   # Optional: LLM parameters
        "tool_choice": "auto",
        "temperature": 0.7
    }
)
```

### 3. PocketAgent Constructor

PocketAgent initialization parameters:

```python
agent = PocketAgent(
    agent_config,           # Required: AgentConfig instance
    mcp_config=None,        # Optional: MCP server configuration (dict or FastMCP instance)
    router=None,            # Optional: LiteLLM Router instance for rate limiting/fallbacks  
    logger=None,            # Optional: Custom logger instance
    hooks=None,             # Optional: AgentHooks instance for customization
    sub_agents=[],          # Optional: List of PocketAgent instances to use as sub-agents
    **client_kwargs         # Optional: Additional kwargs for PocketAgentClient
)
```

**Parameter Details:**
- **agent_config**: Required `AgentConfig` instance containing model, prompts, etc.
- **mcp_config**: MCP server configuration (can be None if using only sub_agents)
- **router**: LiteLLM Router for advanced model routing and rate limiting
- **logger**: Custom logger for capturing agent activity
- **hooks**: `AgentHooks` instance for customizing behavior at execution points
- **sub_agents**: List of `PocketAgent` instances that become callable tools
- **client_kwargs**: Additional parameters passed to `PocketAgentClient`

### 4. StepResult

Return value from `step()` method:

```python
@dataclass
class StepResult:
    llm_message: LitellmMessage                                 # LLM response
    tool_execution_results: Optional[list[ToolResult]] = None   # Tool results
```

### 5. AgentHooks

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

### 4. Multi-Agent Orchestration

```python
class MultiAgentOrchestrator(PocketAgent):
    """Orchestrator that coordinates specialized sub-agents"""
    
    def __init__(self, *args, **kwargs):
        # Create specialized sub-agents
        self.research_agent = PocketAgent(
            agent_config=AgentConfig(
                llm_model="gpt-3.5-turbo",
                name="ResearchAgent",
                role_description="Web research and information gathering specialist",
                system_prompt="You are a research specialist. Find and analyze information from web sources."
            ),
            mcp_config=self._get_research_mcp_config()
        )
        
        self.analysis_agent = PocketAgent(
            agent_config=AgentConfig(
                llm_model="gpt-3.5-turbo", 
                name="AnalysisAgent",
                role_description="Data analysis and visualization specialist",
                system_prompt="You are a data analyst. Analyze data and create visualizations."
            ),
            mcp_config=self._get_analysis_mcp_config()
        )
        
        # Initialize orchestrator with sub-agents
        super().__init__(
            sub_agents=[self.research_agent, self.analysis_agent],
            *args, **kwargs
        )
    
    async def run(self):
        """Coordinate sub-agents for complex tasks"""
        await self.add_user_message(
            "Create a comprehensive market analysis report for AI startups. "
            "First research the current market, then analyze the data to identify trends."
        )
        
        # Let the orchestrator coordinate sub-agents automatically
        step_result = await self.step()
        while step_result.llm_message.tool_calls:
            step_result = await self.step()
        
        return {
            "status": "completed",
            "final_report": step_result.llm_message.content,
            "sub_agents_used": [
                r.tool_call_name for r in (step_result.tool_execution_results or [])
                if r.tool_call_name.endswith("-message")
            ]
        }
    
    def _get_research_mcp_config(self):
        return {"mcpServers": {"web_search": {"transport": "stdio", ...}}}
    
    def _get_analysis_mcp_config(self):
        return {"mcpServers": {"data_viz": {"transport": "stdio", ...}}}

# Usage
orchestrator_config = AgentConfig(
    llm_model="gpt-4",
    system_prompt="You are an expert orchestrator who coordinates specialized agents. "
                  "Use ResearchAgent for research tasks and AnalysisAgent for data analysis. "
                  "Combine their outputs into comprehensive reports."
)

orchestrator = MultiAgentOrchestrator(agent_config=orchestrator_config)
result = await orchestrator.run()
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

### 4. Multi-Agent Design

**Do:**
- Use clear, descriptive names and role descriptions for sub-agents
- Design sub-agents with focused, specific capabilities
- Ensure sub-agent `run()` methods accept a message parameter
- Use meaningful system prompts that describe how to coordinate sub-agents
- Consider the execution lock when designing parallel workflows

**Don't:**
- Create overly complex agent hierarchies (keep it simple)
- Pass large amounts of data directly through sub-agent messages
- Forget that sub-agents maintain their own conversation history
- Create circular dependencies between agents

**Sub-Agent Design Pattern:**

```python
class SpecializedSubAgent(PocketAgent):
    async def run(self, message: str = None):
        """Sub-agent must accept message parameter"""
        if not message:
            return "No task provided"
        
        # Reset for each task to avoid context pollution
        self.reset_messages()
        
        await self.add_user_message(message)
        
        step_result = await self.step()
        while step_result.llm_message.tool_calls:
            step_result = await self.step()
        
        # Return clean result
        return step_result.llm_message.content
```

**Orchestrator Design Pattern:**

```python
orchestrator_config = AgentConfig(
    llm_model="gpt-4",  # Use more capable model for coordination
    name="Orchestrator",
    system_prompt=(
        "You coordinate specialized agents to complete complex tasks. "
        "Available agents:\n"
        "- ResearchAgent: For web research and information gathering\n"
        "- AnalysisAgent: For data analysis and visualization\n"
        "- WritingAgent: For document creation and editing\n\n"
        "Break down complex tasks and delegate appropriately."
    )
)
```

### 5. Hook Usage

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

### 5. Multi-Agent Research and Analysis Workflow

```python
from pocket_agent import PocketAgent, AgentConfig

class ResearchAgent(PocketAgent):
    """Specialized agent for web research and information gathering"""
    
    async def run(self, message: str = None):
        if not message:
            return "No research task provided"
        
        # Start fresh for each research task
        self.reset_messages()
        await self.add_user_message(f"Research this topic comprehensively: {message}")
        
        step_result = await self.step()
        while step_result.llm_message.tool_calls:
            step_result = await self.step()
        
        return step_result.llm_message.content

class AnalysisAgent(PocketAgent):
    """Specialized agent for data analysis and insights"""
    
    async def run(self, message: str = None):
        if not message:
            return "No analysis task provided"
        
        self.reset_messages()
        await self.add_user_message(f"Analyze the following data and provide insights: {message}")
        
        step_result = await self.step()
        while step_result.llm_message.tool_calls:
            step_result = await self.step()
        
        return step_result.llm_message.content

class ReportAgent(PocketAgent):
    """Specialized agent for creating comprehensive reports"""
    
    async def run(self, message: str = None):
        if not message:
            return "No report content provided"
        
        self.reset_messages()
        await self.add_user_message(f"Create a comprehensive report from this information: {message}")
        
        step_result = await self.step()
        while step_result.llm_message.tool_calls:
            step_result = await self.step()
        
        return step_result.llm_message.content

class ComprehensiveResearchSystem(PocketAgent):
    """Orchestrator that coordinates research, analysis, and reporting"""
    
    def __init__(self, research_topic, *args, **kwargs):
        self.research_topic = research_topic
        
        # Create specialized sub-agents
        research_agent = ResearchAgent(
            agent_config=AgentConfig(
                llm_model="gpt-3.5-turbo",
                name="ResearchAgent",
                role_description="Web research and information gathering specialist",
                system_prompt="You are a thorough researcher. Find comprehensive information from web sources and provide detailed findings."
            ),
            mcp_config={"mcpServers": {"web_search": {"transport": "stdio", "command": "python", "args": ["web_search_server.py"]}}}
        )
        
        analysis_agent = AnalysisAgent(
            agent_config=AgentConfig(
                llm_model="gpt-3.5-turbo",
                name="AnalysisAgent", 
                role_description="Data analysis and insights specialist",
                system_prompt="You are a data analyst. Analyze information and identify key trends, patterns, and insights."
            ),
            mcp_config={"mcpServers": {"data_analysis": {"transport": "stdio", "command": "python", "args": ["analysis_server.py"]}}}
        )
        
        report_agent = ReportAgent(
            agent_config=AgentConfig(
                llm_model="gpt-4",  # Use more capable model for final report
                name="ReportAgent",
                role_description="Comprehensive report creation specialist",
                system_prompt="You are an expert report writer. Create well-structured, comprehensive reports with clear sections and professional formatting."
            ),
            mcp_config={"mcpServers": {"document_gen": {"transport": "stdio", "command": "python", "args": ["doc_server.py"]}}}
        )
        
        # Initialize with sub-agents
        super().__init__(
            agent_config=AgentConfig(
                llm_model="gpt-4",
                name="ResearchOrchestrator", 
                system_prompt=(
                    "You are a research orchestrator that coordinates specialized agents:\n"
                    "1. ResearchAgent: Use for gathering information and research\n"
                    "2. AnalysisAgent: Use for analyzing data and identifying insights\n"
                    "3. ReportAgent: Use for creating final comprehensive reports\n\n"
                    "Always coordinate these agents in logical order: Research → Analysis → Report"
                )
            ),
            sub_agents=[research_agent, analysis_agent, report_agent],
            *args, **kwargs
        )
    
    async def run(self):
        """Execute comprehensive research workflow"""
        await self.add_user_message(
            f"Create a comprehensive research report on '{self.research_topic}'. "
            f"First gather research, then analyze the findings, and finally create a professional report."
        )
        
        # Let the orchestrator coordinate the workflow
        step_result = await self.step()
        while step_result.llm_message.tool_calls:
            step_result = await self.step()
        
        return {
            "status": "completed",
            "research_topic": self.research_topic,
            "final_report": step_result.llm_message.content,
            "agents_utilized": [
                r.tool_call_name for r in (step_result.tool_execution_results or [])
                if r.tool_call_name.endswith("-message")
            ]
        }

# Usage Example
async def run_comprehensive_research():
    research_system = ComprehensiveResearchSystem(
        research_topic="Impact of AI on Healthcare in 2024"
    )
    
    result = await research_system.run()
    
    print("Research completed!")
    print(f"Topic: {result['research_topic']}")
    print(f"Agents used: {result['agents_utilized']}")
    print(f"Final report:\n{result['final_report'][:500]}...")
    
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

### 2. Multi-Agent Debugging

**Issue: Sub-agents not being called**
```python
# Check if sub-agents are properly registered as tools
tools = await main_agent.mcp_client.get_tools(format="openai")
sub_agent_tools = [t for t in tools if t.get('function', {}).get('name', '').endswith('-message')]
print(f"Available sub-agent tools: {[t['function']['name'] for t in sub_agent_tools]}")

# Verify sub-agents are configured
print(f"Sub-agents: {[agent.agent_config.name for agent in main_agent.sub_agents if hasattr(main_agent, 'sub_agents')]}")
```

**Issue: Sub-agent execution errors**
```python
# Debug sub-agent execution with detailed logging
class MultiAgentDebugHooks(AgentHooks):
    async def pre_tool_call(self, context: HookContext, tool_call):
        if tool_call.name.endswith('-message'):
            print(f"🤖 Calling sub-agent: {tool_call.name}")
            print(f"   Message: {tool_call.arguments.get('message', 'N/A')}")
        return tool_call
    
    async def post_tool_call(self, context: HookContext, tool_call, result):
        if tool_call.name.endswith('-message'):
            print(f"✅ Sub-agent {tool_call.name} completed")
            print(f"   Result preview: {str(result.tool_result_content)[:100]}...")
        return result

# Use debug hooks with your orchestrator
orchestrator = PocketAgent(
    agent_config=config,
    sub_agents=sub_agents,
    hooks=MultiAgentDebugHooks()
)
```

**Issue: Sub-agent context pollution**
```python
# Monitor sub-agent message history
class SubAgentMonitor(AgentHooks):
    async def pre_tool_call(self, context: HookContext, tool_call):
        if tool_call.name.endswith('-message'):
            # Find the sub-agent being called
            agent_name = tool_call.name.replace('-message', '')
            sub_agent = next((a for a in context.agent.sub_agents if a.agent_config.name == agent_name), None)
            if sub_agent:
                print(f"Sub-agent {agent_name} has {len(sub_agent.messages)} messages in history")
        return tool_call
```

### 3. Debugging Hooks

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

### 4. Multi-Agent Systems

Pocket Agent supports multi-agent architectures where you can compose agents by passing other PocketAgent instances as sub-agents. Sub-agents are automatically converted to MCP tools that the main agent can call.

#### Basic Multi-Agent Setup

```python
from pocket_agent import PocketAgent, AgentConfig

# Create specialized sub-agents
research_agent_config = AgentConfig(
    llm_model="gpt-3.5-turbo",
    name="ResearchAgent", 
    role_description="Specialized in web research and information gathering",
    system_prompt="You are a research specialist. Find and analyze information from web sources."
)

research_agent = PocketAgent(
    agent_config=research_agent_config,
    mcp_config=research_mcp_config  # MCP config with research tools
)

analysis_agent_config = AgentConfig(
    llm_model="gpt-3.5-turbo",
    name="AnalysisAgent",
    role_description="Specialized in data analysis and visualization", 
    system_prompt="You are a data analyst. Analyze data and create visualizations."
)

analysis_agent = PocketAgent(
    agent_config=analysis_agent_config,
    mcp_config=analysis_mcp_config  # MCP config with analysis tools
)

# Main orchestrator agent with sub-agents
main_config = AgentConfig(
    llm_model="gpt-4",
    name="Orchestrator",
    system_prompt="You coordinate between specialized agents to complete complex tasks. Use ResearchAgent for research tasks and AnalysisAgent for data analysis."
)

orchestrator = PocketAgent(
    agent_config=main_config,
    mcp_config=None,  # mcp_config can be None if only using sub-agents
    sub_agents=[research_agent, analysis_agent]
)
```

#### Sub-Agent Tool Integration

When you add sub-agents to a main agent, they are automatically exposed as tools with names formatted as `{agent_name}-message` (e.g., "ResearchAgent-message", "AnalysisAgent-message"). Each sub-agent tool has a single `message: str` argument.

```python
# The main agent can now call sub-agents as tools:
class OrchestratorAgent(PocketAgent):
    async def run(self):
        await self.add_user_message("Research the latest developments in AI and then analyze the trends")
        
        step_result = await self.step()
        while step_result.llm_message.tool_calls:
            step_result = await self.step()
        
        return {"result": step_result.llm_message.content}

# When the orchestrator calls "ResearchAgent-message" with message="Research latest AI developments"
# It automatically invokes research_agent.run("Research latest AI developments")
```

#### Sub-Agent Implementation Requirements

**Important**: Sub-agents must implement their `run` method to:
1. Accept a single string argument (the message from the tool call)
2. Return one of: `None`, `str`, `dict`, or FastMCP `ToolResult` instance

```python
class ResearchSubAgent(PocketAgent):
    async def run(self, message: str = None):
        """Sub-agent run method must accept message parameter"""
        if message:
            await self.add_user_message(message)
        
        step_result = await self.step()
        while step_result.llm_message.tool_calls:
            step_result = await self.step()
        
        # Return string result that will be passed back to main agent
        return step_result.llm_message.content
```

#### Complex Multi-Agent Coordination

```python
class ComplexCoordinatorAgent(PocketAgent):
    async def run(self):
        """Coordinate multiple sub-agents for complex workflow"""
        await self.add_user_message("Create a comprehensive market analysis report for AI startups")
        
        # Let the orchestrator decide how to use sub-agents
        step_result = await self.step()
        while step_result.llm_message.tool_calls:
            step_result = await self.step()
        
        return {
            "status": "completed",
            "report": step_result.llm_message.content,
            "sub_agents_used": [
                result.tool_call_name for result in (step_result.tool_execution_results or [])
                if result.tool_call_name.endswith("-message")
            ]
        }
```

#### Sub-Agent Execution Model

- **Thread Safety**: Sub-agents execute within a lock, preventing parallel calls to the same sub-agent instance
- **Isolation**: Each sub-agent maintains its own conversation history and state
- **Tool Composition**: Sub-agents can have their own MCP tools, and main agents can have both sub-agents and regular MCP tools
- **Error Handling**: Sub-agent errors are handled through the standard tool error handling system

#### Multi-Layer Agent Hierarchies

```python
# You can create hierarchies of agents
level2_specialist = PocketAgent(
    agent_config=AgentConfig(
        llm_model="gpt-3.5-turbo",
        name="DataProcessor",
        system_prompt="Process and clean data"
    ),
    mcp_config=data_processing_config
)

level1_analyst = PocketAgent(
    agent_config=AgentConfig(
        llm_model="gpt-3.5-turbo", 
        name="Analyst",
        system_prompt="Analyze data using specialized processing tools"
    ),
    sub_agents=[level2_specialist],
    mcp_config=analysis_config
)

top_level_coordinator = PocketAgent(
    agent_config=AgentConfig(
        llm_model="gpt-4",
        name="Coordinator",
        system_prompt="Coordinate complex analysis workflows"
    ),
    sub_agents=[level1_analyst],
    mcp_config=coordination_config
)
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

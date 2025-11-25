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
- [Quick Start](#quick-start)
- [Documentation](#-documentation)

---

## Why Pocket Agent?

Most agent frameworks are severely over-bloated. The reason for this is that they are trying to support too many things at once and make every possible agent implementation "simple". In the end, you might find yourself using 5-10% of the framework's code for your use case, meaning 90-95% is noise when you encounter a bug. This makes adding features and debugging unnecessarily complex.

Pocket Agent takes the opposite approach by implementing the basic functionalities of an Agent as an abstraction one layer above the MCP Client with useful features to provide first-class support for customization. This makes Pocket Agent extremely lightweight. In fact, so lightweight that the **entire core framework and documentation can easily fit into most LLM context windows** (see the built-in utility for this [here](#quick-start-for-llms)). The minimal nature also means that you will use 90-95% of the framework's code for every agent you create.

Despite the minimal code, Pocket Agent is extremely powerful, even out-of-the-box with no custom extensions. By default, Pocket Agent enables users to:
- Use any remote or local MCP server(s) to integrate any amount of custom tools
- Run a typical agent loop (generate message, execute tool calls, handle tool results until the LLM does not call any tools)
- Allow tools to be run in parallel, minimizing execution bottlenecks
- Build multi-agent systems using built-in features to facilitate interaction
- View agents' behavior in a minimal built-in CLI frontend
- Use any local or remote LLM and enforce tpm/rpm limits for parallel LLM generations (Thanks to [LiteLLM](https://docs.litellm.ai/))


---


## Design Principles

### 🚀 **Lightweight & Simple**
- Clean abstractions that separate agent logic from MCP client details
- Core functionality is only 2 files
- Only implements the core, non-case-specific features of an LLM agent

### 💡 **Extensible**
- Easily integrate a custom frontend using the built-in event system
- Easily implement fully custom agent behaviors
- Easily develop multi-agent systems


---


## Installation

Install with uv (Recommended):
```bash
uv add pocket-agent
```

Install with pip:
```bash
pip install pocket-agent
```

## Quick Start (For LLMs)

LLMs are excellent at building with Pocket Agent. Mostly, because the entirety of the core framework is minimal enough to easily fit into most LLM context windows. 

Pocket Agent provides a simple command-line utility for this exact use case: **`pocket-dump`**

### 📄 Using pocket-dump

After installation, use `pocket-dump` to generate complete documentation with source code:

```bash
# Generate complete documentation + source code as pocket-agent-source.md
pocket-dump

# Custom filename
pocket-dump my-pocket-agent-reference.md

# Only include documentation
pocket-dump --docs-only

# Only include source code  
pocket-dump --source-only
```

The generated file contains everything an LLM needs to understand and build with Pocket Agent - no need to piece together multiple documentation sources or guess at implementation details.

## Quick Start (For Humans)

You can use PocketAgent out of the box in 3 steps:

1. Configure the agent:
    ```python
    from pocket_agent import AgentConfig

    # Configure and run
    config = AgentConfig(
        llm_model="gpt-4",
        system_prompt="You are a helpful assistant."
    )

2. Define the mcp servers the agent can access:
    ```python
    servers = {
        "mcpServers": {
            "weather_tools": {
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
                "cwd": "path/to/server.py"
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

In the above example, the `run` method will add the input as a user message, and then run in a loop generating new messages and running any tools called by the configured LLM until the configured LLM generates a response which does not have any tool calls. All of the messages to and from the agent are neatly formatted in the console by default.

**Pocket Agents are meant to be extended far beyond this behavior. Refer to the sections below for implementation [examples](./cookbook/) and [full documentation](pocket_agent/docs/)**

## 🧑‍🍳 [Cookbook](./cookbook/)
Find example implementations of Pocket Agents and their extensions in the [Cookbook](./cookbook/)

## 📖 Documentation

For comprehensive documentation on all features and advanced usage, see the [documentation folder](pocket_agent/docs/):

- **[Getting Started](pocket_agent/docs/01_getting-started.md)** - Detailed quick start guide and examples
- **[Core Concepts](pocket_agent/docs/02_core-concepts.md)** - PocketAgent base class, step method, message management
- **[Hooks and Events](pocket_agent/docs/03_hooks-and-events.md)** - Customizing agent behavior and frontend integration
- **[Multi-Model Support](pocket_agent/docs/04_multi-model-support.md)** - Working with different LLM providers
- **[PocketAgentClient](pocket_agent/docs/05_client.md)** - MCP client wrapper and advanced features
- **[Multi-Agent Systems](pocket_agent/docs/06_multi-agent.md)** - Building complex agent architectures
- **[Agent-as-a-Server](pocket_agent/docs/07_agent-as-a-server.md)** - Using agents as MCP servers
- **[Testing](pocket_agent/docs/08_testing.md)** - Test suite and coverage


# Feature Roadmap

## Core Features
| Feature | Status | Priority | Description |
|---------|--------|----------|-------------|
| **Agent Abstraction** | ✅ Implemented | - | Basic agent abstraction with PocketAgent base class |
| **MCP Protocol Support** | ✅ Implemented | - | Full integration with Model Context Protocol via fastmcp |
| **Multi-Model Support** | ✅ Implemented | - | Support for any LiteLLM compatible model/endpoint |
| **Tool Execution** | ✅ Implemented | - | Automatic parallel tool calling and results handling |
| **Hook System** | ✅ Implemented | - | Allow configurable hooks to inject functionality during agent execution |
| **Logging Integration** | ✅ Implemented | - | Built-in logging with custom logger support |
| **Multi-Agent Integration** | ✅ Implemented | - | Allow a PocketAgent to accept other PocketAgents as Sub Agents and automatically set up Sub Agents as tools for the Agent to use |
| **function-as-a-tool support** | 📋 Planned | Medium | Allow python functions to be passed to PocketAgent to act as tools |
| **Define Defaults for standard MCP Client handlers | 📋 Planned | Medium | Standard MCP client methods (i.e. sampling, progress, etc) may benefit from default implementations if custom behavior is not often needed |
| **Streaming Responses** | 📋 Planned | Medium | Real-time response streaming support |
| **Define Defaults for standard MCP Client handlers | 📋 Planned | Medium | Standard MCP client methods (i.e. sampling, progress, etc) may benefit from default implementations if custom behavior is not often needed |
| **Resources Integration** | 📋 Planned | Medium | Automatically set up mcp read_resource functionality as a tool (resources are not very commonly used today so this may not be necessary) |

## Modality support
| Modality | Status | Priority | Description |
|---------|--------|----------|-------------|
| **Text** | ✅ Implemented | - | Multi-modal input support for vision models |
| **Images** | ✅ Implemented | - | Multi-modal input support for VLMs with option to enable/disable |
| **Audio** | 📋 Planned | Low | Multi-modal input support for LLMs which allow audio inputs |



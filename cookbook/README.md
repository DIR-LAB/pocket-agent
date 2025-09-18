# Pocket Agent Cookbook

<div align="center">

# Pocket-Agent Cookbook

<img src="./assets/cooking.png" alt="Pocket Agent" width="300" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">


</div>

This cookbook contains practical examples and reference implementations for the pocket_agent framework. Each example demonstrates different use cases, patterns, and capabilities.
## Quick Start

1. **Clone this repository:**
    ```bash
    git clone https://github.com/DIR-LAB/pocket-agent
    cd pocket-agent
    ```

2. **Set openai api key**:
    ```bash
    export OPENAI_API_KEY="your-openai-key"
    ```

2. **Sync with the cookbook dependencies:**
    - For all cookbook examples:
        ```bash
        uv sync --group cookbook-all
        ```
    - For specific examples, refer to the UV group listed in the [Executable Examples](#executable-examples). If no group is indicated, the example does not require special dependencies.
    

3. **Run any of the examples using the entry points listed below in the [Executable Examples](#executable-examples):**

    ```bash
    # Running the Simple Chat Agent example
    uv run cookbook/agent_examples/01_simple_chat_agent/agent.py
    #      [#######👆 Entry Point from examples below👆#######]

    ```


## Executable Examples

### Agent Implementation Patterns
- **[Simple Chat Agent](agent_examples/01_simple_chat_agent/)** - Basic agent implementations using the minimal, built-in CLI interface
    - Weather Agent:
        - Entry point: `cookbook/agent_examples/01_simple_chat_agent/agent.py`
    - RAG Agent:
        - **UV group:** `cookbook-rag`
        - Entry point: `cookbook/agent_examples/01_simple_chat_agent/rag_agent.py`
    

- **[Multi-Agent](agent_examples/02_multi_agent/)** - Simple multi-agent example including a SimpleAgent and a WeatherAgent
    - Entry point: `cookbook/agent_examples/02_multi_agent/agents.py`

- **[Agent-as-a-Server](agent_examples/03_agent_as_a_server/)** - Simple multi-agent example including a SimpleAgent and a WeatherAgent
    - Entry point: `cookbook/agent_examples/03_agent_as_a_server/agent.py`

### Adding a Frontend
- **[Rich Console Output](event_hook_examples/01_rich_console_output/)** - Create a more expressive CLI frontend using Rich
    - **UV group:** `cookbook-rich`
    - Single Agent example:
        - Entry point: `cookbook/event_hook_examples/01_rich_console_output/agent.py`
    - Multi-Agent example:
        - Entry point: `cookbook/event_hook_examples/01_rich_console_output/multi_agent.py`

- **[Streamlit UI](event_hook_examples/02_streamlit_chat_frontend/)** - Create a dedicated UI using Streamlit
    - **UV group:** `cookbook-streamlit`
    - Entry point: `streamlit run cookbook/event_hook_examples/02_streamlit_chat_frontend/frontend.py`


## Server Examples


## Sample LiteLLM Router
- An example of a LiteLLM router can be found in [sample_router.py](./sample_router.py). More documentation on LiteLLM Routers can be found [here](https://docs.litellm.ai/docs/routing).













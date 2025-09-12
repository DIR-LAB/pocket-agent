# Pocket Agent Cookbook

This cookbook contains practical examples and reference implementations for the pocket_agent framework. Each example demonstrates different use cases, patterns, and capabilities.

## Examples Overview

### Creating a simple calculator agent
- **[01_simple_chat_agent](01_simple_chat_agent/)** - Simple CLI agent with a weather server
    - Entry point: `cookbook/01_simple_chat_agent/agent.py`

### Adding a Frontend to your agent
- **[02_streamlit_chat_frontend](02_streamlit_chat_frontend/)** - Interactive Streamlit web UI with real-time event system integration
     - Entry point: `streamlit run cookbook/02_streamlit_chat_frontend/frontend.py`

## Quick Start

1. **Clone this repository:**
    ```bash
    git clone https://github.com/DIR-LAB/pocket-agent
    cd pocket-agent
    ```

2. **eSet openai api key**:
    ```bash
    export OPENAI_API_KEY="your-openai-key"
    ```

2. **Sync with the cookbook dependencies:**

    ```bash
    uv sync --group cookbook
    ```

3. **Run any of the examples using the entry points listed above:**

    ```bash
    # Example 01
    uv run cookbook/01_simple_chat_agent/agent.py

    # Example 02
    uv run streamlit run cookbook/02_streamlit_chat_frontend/frontend.py
    ```



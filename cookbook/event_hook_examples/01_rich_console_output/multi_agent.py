#!/usr/bin/env python3
"""
Example showing how to use the Rich console formatter for beautiful output
"""

import asyncio
import json
from typing import Dict, Any
from pocket_agent.agent import PocketAgent, AgentConfig
from rich_hooks import RichAgentHooks

# Optional: Create a custom Rich console with specific settings
try:
    from rich.console import Console
    from rich.theme import Theme
    
    # Define a custom theme
    custom_theme = Theme({
        "user": "bold bright_blue",
        "assistant": "bold bright_green", 
        "tool": "bold magenta",
        "error": "bold red"
    })
    
    # Create console with custom theme
    console = Console(theme=custom_theme, width=120)
    RICH_AVAILABLE = True
    
except ImportError:
    console = None
    RICH_AVAILABLE = False
    print("Rich not available, will use fallback formatting")



#########################################################
# Simple Agent
#########################################################
class SimpleAgent(PocketAgent):
    """Simple agent that only implements the run method"""
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """
        Simple conversation loop for interactive chat.
        This method handles a single user input and generates a response.
        """
        await self.add_user_message(user_input)
        # Generate response and execute tools in a loop
        step_result = await self.step()
        while step_result.llm_message.tool_calls is not None:
            step_result = await self.step()
        
        return step_result.llm_message.content



async def main():
    import os
    """Demonstrate Rich formatting with a simple agent"""
        # Create agent with Rich hooks
    if RICH_AVAILABLE:
        hooks = RichAgentHooks(console=console)
        print("🎨 Using Rich formatting!")
    else:
        raise ImportError("Rich is not available")

    

    sub_agent_config = AgentConfig(
        llm_model="gpt-5-nano",
        name="WeatherReporter",
        role_description="A sub-agent that helps with specific tasks",
        system_prompt="You are a helpful weather reporter who uses the weather tools to get the weather information"
    )

    # Create MCP config (using a simple weather server as example)
    sub_agent_mcp_config = {
        "mcpServers": {
            "weather": {
                "command": "python",
                "args": ["server.py"],
                "cwd": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "servers", "simple_weather")
            }
        }
    }

    weather_agent = SimpleAgent(
        agent_config=sub_agent_config,
        mcp_config=sub_agent_mcp_config,
        hooks=hooks
    )


    # Create agent config
    primary_agent_config = AgentConfig(
        llm_model="gpt-5-nano",
        name="MainAgent",
        system_prompt="""You are a helpful assistant that coordinates with other agents and formats responses for optimal visual display.

FORMATTING GUIDELINES:
1. **Use Markdown formatting** for better readability:
   - Use **bold** and *italic* for emphasis
   - Use # Headers for sections
   - Use - bullet points for lists
   - Use numbered lists: 1. First item
   - Use `code` for inline code
   - Use ```language code blocks for multi-line code

2. **For structured data**, always use proper JSON formatting:
   ```json
   {
     "key": "value",
     "nested": {
       "data": "formatted nicely"
     }
   }
   ```

3. **For code examples**, use appropriate language tags:
   ```python
   def example_function():
       return "properly formatted"
   ```

4. **Organize responses** with clear sections and spacing.

5. **When explaining complex topics**, use headers and bullet points to break down information.

Remember: Your responses will be displayed in a beautiful console interface, so good formatting enhances the user experience significantly.

GUIDELINES FOR INTERACTING WITH OTHER AGENTS:
1. Use the `{agent_name}-message` tool to interact with other agents.
2. If there are multiple related tasks to give to an agent, combine them into a single message to the agent.
"""
    )

    agent = SimpleAgent(
        agent_config=primary_agent_config,
        hooks=hooks,
        sub_agents=[weather_agent]
    )

    await agent.run("Hello! what's the weather in Sydney?")
    await agent.run("get me the 3 day forecast for Sydney, Tokyo, and London and make them into a table")
    await agent.run("Thank you! Can you also tell me a joke?")  

if __name__ == "__main__":
    asyncio.run(main())
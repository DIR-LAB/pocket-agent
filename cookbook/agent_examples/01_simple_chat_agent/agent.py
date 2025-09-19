import asyncio
from pocket_agent import PocketAgent, AgentConfig
from typing import Dict, Any


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

    mcp_config = {
        "mcpServers": {
            "weather": {
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
                "cwd": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "servers", "simple_weather")
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
    
    await agent.run("Hello! What is the weather in Tokyo?")
    await agent.run("Get the 3 day forecast for tokyo, sydney and london and make them into a table")

if __name__ == "__main__":
    asyncio.run(main())
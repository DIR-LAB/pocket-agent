import asyncio
from pocket_agent import PocketAgent, AgentConfig
import litellm

# Suppress LiteLLM's verbose logging
litellm.set_verbose = False
litellm.log_level = "WARNING"

class SimpleAgent(PocketAgent):
    async def run(self):
        """Simple conversation loop"""

        while True:
            user_input = input("Your input: ")
            if user_input.lower() == 'quit':
                break
                
            # Add user message
            await self.add_user_message(user_input)
            
             # Generates response and executes any tool calls
            step_result = await self.step()
            while step_result.llm_message.tool_calls is not None:
                step_result = await self.step()
    
        return {"status": "completed"}



async def main():
    import os

    mcp_config = {
        "mcpServers": {
            "weather": {
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
                "cwd": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "servers", "simple_weather")
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

if __name__ == "__main__":
    asyncio.run(main())
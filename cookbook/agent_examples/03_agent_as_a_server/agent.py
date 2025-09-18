import asyncio
import os
from pocket_agent import PocketAgent


#########################################################
# Simple Agent (to be used as the main agent)
#########################################################
class SimpleAgent(PocketAgent):

    # run method of main agents does not need to accept any arguments
    async def run(self) -> dict:

        # main agent will execute until the user quits
        while True:
            user_input = input("Your input: ")
            if user_input.lower() == 'quit':
                break
                
            # Add user message to the agent
            await self.add_user_message(user_input)
            
            # agent will execute until it does not call any tools anymore
            step_result = await self.step()
            while step_result.llm_message.tool_calls is not None:
                step_result = await self.step()
    
        return {"status": "completed"}



#########################################################
# Initialize the simple agent
#########################################################
def create_simple_agent():
    # MCP config for the simple agent to give it access to the utilities server's tools and the weather agent running as a server
    simple_agent_mcp_config = {
        "mcpServers": {
            "utilities": {
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
                "cwd": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "servers", "simple_utilities")
            },
            "weather": {
                "transport": "stdio",
                "command": "python",
                "args": ["agent_server.py"],
                "cwd": os.path.join(os.path.dirname(os.path.abspath(__file__)))
            }
        }
    }

    # Agent config for the simple agent
    simple_agent_config = AgentConfig(
        llm_model="gpt-5-nano",
        system_prompt="You are a helpful assistant who answers user questions and uses provided tools when applicable"
    )

    # Create and return the simple agent instance
    simple_agent = SimpleAgent(
        agent_config=simple_agent_config,
        mcp_config=simple_agent_mcp_config
    )
    return simple_agent


async def main():
    simple_agent = create_simple_agent()
    await simple_agent.run()


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import os

from pocket_agent import PocketAgent, AgentConfig



#########################################################
# Weather Agent (to be used as a sub-agent)
#########################################################
class WeatherAgent(PocketAgent):
    """Simple agent that only implements the run method"""

    # run method of sub-agents must accept a single string argument
    async def run(self, user_input: str) -> str:
        await self.add_user_message(user_input)

        # agent will execute until it does not call any tools anymore
        step_result = await self.step()
        while step_result.llm_message.tool_calls is not None:
            step_result = await self.step()
        
        # agent returns the final message content as its result
        return step_result.llm_message.content



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
# Initialize the weather agent
#########################################################
def create_weather_agent():
    # Agent config for the weather agent
    weather_agent_config = AgentConfig(
        llm_model="gpt-5-nano",
        name="Weather_Reporter",
        role_description="provide accurate weather information for cities",
        system_prompt="You are a weather reporter who answers user questions and uses the weather tools when applicable"
    )

    # MCP config for the weather agent
    weather_mcp_config = {
        "mcpServers": {
            "weather": {
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
                "cwd": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "servers", "simple_weather")
            }
        }
    }

    # Create and return the weather agent instance
    weather_agent = WeatherAgent(
        agent_config=weather_agent_config,
        mcp_config=weather_mcp_config
    )
    return weather_agent


#########################################################
# Initialize the simple agent
#########################################################
def create_simple_agent(sub_agents: list[PocketAgent]):
    # MCP config for the simple agent to give it access to the utilities server's tools
    # Note: since the simple agent will have a sub-agent, it is not required to pass a mcp config
    simple_agent_mcp_config = {
        "mcpServers": {
            "utilities": {
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
                "cwd": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "servers", "simple_utilities")
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
        mcp_config=simple_agent_mcp_config,
        sub_agents=sub_agents
    )
    return simple_agent


async def main():

    # Create the agents
    weather_agent = create_weather_agent()
    simple_agent = create_simple_agent([weather_agent])

    # Run the simple agent
    await simple_agent.run()

if __name__ == "__main__":
    asyncio.run(main())
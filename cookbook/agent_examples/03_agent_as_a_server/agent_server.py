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


if __name__ == "__main__":
    weather_agent = create_weather_agent()
    server = weather_agent.as_mcp_server()
    server.run()
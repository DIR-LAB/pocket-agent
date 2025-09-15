import asyncio
from pocket_agent import PocketAgent
from typing import Dict, Any


class SimpleAgent(PocketAgent):
    """Simple agent that only implements the run method"""
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """
        Simple conversation loop for interactive chat.
        This method handles a single user input and generates a response.
        """
        self.add_user_message(user_input)
        # Generate response and execute tools in a loop
        step_result = await self.step()
        while step_result.llm_message.tool_calls is not None:
            step_result = await self.step()
        
        return step_result.llm_message.content
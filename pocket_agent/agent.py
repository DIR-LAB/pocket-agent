import litellm
from litellm import Router
from abc import abstractmethod
from fastmcp.client.logging import LogMessage
import uuid
from typing import Optional
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Union


from pocket_agent.client import Client

### Create default logger ###
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)



@dataclass
class GenerateMessageResult:
    message_content: Optional[str]
    image_base64s: Optional[list[str]]
    reasoning_content: Optional[str]
    thinking_blocks: Optional[list[dict]]
    tool_calls: Optional[list[dict]]


@dataclass
class AgentEvent:
    event_type: str
    data: dict


@dataclass
class AgentConfig:
    """Configuration class to make agent setup cleaner"""
    llm_model: str
    agent_id: Optional[str] = None
    context_id: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: Optional[list[dict]] = None
    allow_images: bool = False
    completion_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "tool_choice": "auto"
        # we can add more llm config items if needed
    })

    def get_completion_kwargs(self) -> Dict[str, Any]:
        # safely get completion kwargs
        return self.completion_kwargs or {}


class MCPAgent:
    def __init__(self, Router: Router,
                 mcp_server_config: dict,
                 agent_config: AgentConfig,
                 logger: Optional[logging.Logger] = None,
                 on_event: Optional[Callable[[str, dict], None]] = None):
        

        self.logger = logger or logging.getLogger("pocket_agent")
        self.context_id = agent_config.context_id or str(uuid.uuid4())
        self.agent_id = agent_config.agent_id or str(uuid.uuid4())
        self.Router = Router
        self.agent_config = agent_config
        self.mcp_client = self._init_client(mcp_server_config)
        self.system_prompt = agent_config.system_prompt or ""
        self.messages = agent_config.messages or []
        self.on_event = on_event
        self.logger.info(f"Initializing MCPAgent with agent_id={self.agent_id}, context_id={self.context_id}, model={agent_config.llm_model}")
        self.logger.info(f"MCPAgent initialized successfully with {len(self.messages)} initial messages")


    def _init_client(self, mcp_server_config: dict):
        """
        Initialize the most basic MCP client with the given configuration.
        Override this to add custom client handlers. More docs can be found here:
         - Elicitation handler: https://gofastmcp.com/clients/elicitation
         - Progress handler: https://gofastmcp.com/clients/progress
         - Sampling handler: https://gofastmcp.com/clients/sampling
         - Message handler: https://gofastmcp.com/clients/message
         - Logging handler: https://gofastmcp.com/clients/logging (pass as mcp_log_handler to Client)
        """
        return Client(mcp_server_config,
                     on_tool_error=self._on_tool_error)


    def _on_tool_error(self, error: Exception) -> Union[False, str]:
        """
        Implement this to handle specific known tool errors.
        Returns False if execution should not continue, str if execution should continue.
        str should be the message to add to the message history.
        """
        return False
    
    def _before_step(self) -> None:
        """
        Implement this to do something before each step.
        """
        pass

    def _after_step(self) -> None:
        """
        Implement this to do something after each step.
        """
        pass


    def _format_messages(self) -> list[dict]:
        # format system prompt and messages in proper format
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.messages
        ]
        return messages


    async def _get_llm_response(self, **override_completion_kwargs) -> dict:
        self.logger.debug(f"Requesting LLM response with model={self.agent_config.llm_model}, message_count={len(self.messages) + 1}")
        # get a response from the llm
        kwargs = self.agent_config.get_completion_kwargs()
        kwargs.update(override_completion_kwargs)
        messages = self._format_messages()
        tools = await self.mcp_client._get_openai_tools()
        kwargs.update({
            "tools": tools,
        })
        try:
            self.logger.debug(f"Requesting LLM response with kwargs={kwargs}")
            response = await self.Router.acompletion(
                model=self.agent_config.llm_model,
                messages=messages,
                **kwargs
            )
            
            self.logger.debug(f"LLM response received: full_response={response}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {e}")
            raise
    


    def add_message(self, message: dict) -> None:
        self.logger.debug(f"Adding message: {message}")
        if self.on_event:
            self.on_event(
                AgentEvent(
                    event_type="new_message",
                    data=message
                )
            )
        self.messages.append(message)


    async def generate(self, **override_completion_kwargs) -> GenerateMessageResult:
        try:
            self.logger.debug("Starting message generation")
            model_response = await self._get_llm_response(**override_completion_kwargs)
            model_new_message = model_response.choices[0].message
            self.add_message(model_new_message.model_dump())

            # Extract fields with simple defaults
            message_content = getattr(model_new_message, 'content', None)
            image_base64s = getattr(model_new_message, 'images', None)
            reasoning_content = getattr(model_new_message, 'reasoning_content', None)
            thinking_blocks = getattr(model_new_message, 'thinking_blocks', None)
            tool_calls = getattr(model_new_message, 'tool_calls', None)
            
            self.logger.debug(f"Generated message with content={bool(message_content)}, tools={bool(tool_calls)}")
            return GenerateMessageResult(
                message_content=message_content,
                image_base64s=image_base64s,
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks,
                tool_calls=tool_calls)
            
        except Exception as e:
            self.logger.error(f"Error in generate method: {e}")
            raise

    

    def _filter_images_from_tool_result_content(self, tool_result_content: list[dict]) -> list[dict]:
        return [content for content in tool_result_content if content["type"] != "image_url"]


    async def step(self, **override_completion_kwargs) -> None:
        self.logger.debug("Starting agent step")
        self._before_step()
        try:
            generate_result = await self.generate(**override_completion_kwargs)
            if generate_result.tool_calls:
                tool_names = [tool_call.get('function', {}).get('name', 'unknown') for tool_call in generate_result.tool_calls]
                self.logger.info(f"Executing {len(generate_result.tool_calls)} tool calls: {tool_names}")

                tool_execution_results = await self.mcp_client.call_tools(generate_result.tool_calls)
                if tool_execution_results:
                    self.logger.debug(f"Received tool execution results: {tool_execution_results}")
                    for tool_execution_result in tool_execution_results:
                        tool_call_id = tool_execution_result.tool_call_id
                        tool_call_name = tool_execution_result.tool_call_name
                        tool_result_content = tool_execution_result.tool_result_content
                        if not self.allow_images:
                            tool_result_content = self._filter_images_from_tool_result_content(tool_result_content)
                        new_message = {
                            "role": "tool",
                            "id": tool_call_id,
                            "name": tool_call_name,
                            "content": tool_result_content
                        }
                        self.add_message(new_message)
                else:
                    self.logger.error("No tool execution results received")
                    raise ValueError("No tool execution results received. Tool calls must have failed silently.")
            else:
                self.logger.debug("No tool calls in generate result")
        finally:
            self._after_step()

    

    def add_user_message(self, user_message: str, image_base64s: Optional[list[str]] = None) -> None:
        image_count = len(image_base64s) if image_base64s else 0
        self.logger.info(f"Adding user message: {user_message} with {image_count} images")
        new_message_content = [
            {
                "type": "text",
                "text": user_message
            }
        ]
        if not self.allow_images:
            image_base64s = None
            if image_base64s:
                # add warning message that images are not allowed
                pass
        else:
            if image_base64s:
                for image_base64 in image_base64s:
                    new_message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    })
        self.add_message({
            "role": "user",
            "content": new_message_content
        })

    
    def reset_messages(self) -> None:
        self.messages = []


    @abstractmethod
    def run(self) -> dict:
        """
        Run the agent.
        Returns the the final result as a dict.
        """
        pass

    @property
    def model(self) -> str:
        return self.agent_config.llm_model

    @property
    def allow_images(self) -> bool:
        return self.agent_config.allow_images


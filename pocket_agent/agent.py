from litellm import Router
import litellm
from litellm.types.utils import (
    ChatCompletionMessageToolCall, 
    Message as LitellmMessage, 
    ModelResponse as LitellmModelResponse
)
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from abc import abstractmethod
from fastmcp.client.logging import LogMessage
import fastmcp
import uuid
import asyncio  # Add this import
from typing import Optional, Tuple
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Union
import json
import traceback

from pocket_agent.client import PocketAgentClient, ToolResult
from pocket_agent.utils.logger import configure_logger as configure_pocket_agent_logger


@dataclass
class AgentEvent:
    event_type: str
    data: dict

@dataclass
class StepResult:
    llm_message: LitellmMessage
    tool_execution_results: Optional[list[ToolResult]] = None


@dataclass
class AgentConfig:
    """Configuration class to make agent setup cleaner"""
    llm_model: str
    agent_id: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: Optional[list[dict]] = None
    allow_images: Optional[bool] = False
    completion_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "tool_choice": "auto"
        # we can add more llm config items if needed
    })

    def get_completion_kwargs(self) -> Dict[str, Any]:
        # safely get completion kwargs
        return self.completion_kwargs or {}


@dataclass
class HookContext:
    """Context object passed to all hooks"""
    agent: 'PocketAgent'
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentHooks:
    """Centralized hook registry with consistent signatures"""
    
    async def pre_step(self, context: HookContext) -> None:
        pass
    
    async def post_step(self, context: HookContext) -> None:
        pass
    
    async def pre_tool_call(self, context: HookContext, tool_call: MCPCallToolRequestParams) -> Optional[MCPCallToolRequestParams]:
        return None
    
    async def post_tool_call(self, context: HookContext, tool_call: MCPCallToolRequestParams, result: ToolResult) -> Optional[ToolResult]:
        return result  # Return modified or None to indicate error

    async def on_llm_response(self, context: HookContext, response: LitellmModelResponse) -> None:
        pass
    
    async def on_event(self, event: AgentEvent) -> None:
        from .utils.console_formatter import ConsoleFormatter
        formatter = ConsoleFormatter()
        formatter.format_event(event)

    #########################################################
    # The following 2 hooks run in the PocketAgentClient
    #########################################################

    # tool result handler will replace the default tool result handler in PocketAgentClient if implemented
    # async def on_tool_result(self, context: HookContext, tool_call: ChatCompletionMessageToolCall, tool_result: FastMCPCallToolResult) -> ToolResult:
    #     pass
    
    async def on_tool_error(self, context: HookContext, tool_call: MCPCallToolRequestParams, error: Exception) -> Union[str, False]:
        if "unexpected_keyword_argument" in str(error):
            tool_call_name = tool_call.name
            tool_format = await context.agent.mcp_client.get_tool_input_format(tool_call_name)
            return "You supplied an unexpected keyword argument to the tool. \
                Try again with the correct arguments as specified in expected format: \n" + json.dumps(tool_format)
        return False



class PocketAgent:
    def __init__(self,
                 agent_config: AgentConfig,
                 mcp_config: dict,
                 router: Router = None,
                 logger: Optional[logging.Logger] = None,
                 hooks: Optional[AgentHooks] = None,
                 **client_kwargs):
        self.agent_id = agent_config.agent_id or str(uuid.uuid4())
        self.logger = logger or configure_pocket_agent_logger()
        if router:
            self.llm_completion_handler = router
        else:
            self.llm_completion_handler = litellm
        self.agent_config = agent_config
        self.hooks = hooks or AgentHooks()
        self.mcp_client = self._init_client(mcp_config, **client_kwargs)
        self.system_prompt = agent_config.system_prompt or ""
        self.messages = agent_config.messages or []
        self.logger.info(f"Initializing MCPAgent with agent_id={self.agent_id}, model={agent_config.llm_model}")
        self.logger.info(f"MCPAgent initialized successfully with {len(self.messages)} initial messages")


    def _init_client(self, mcp_config: dict, **client_kwargs):
        """
        Initialize the most basic MCP client with the given configuration.
        Override this to add custom client handlers.
        """
        # check if on_tool_result is in hooks class
        if hasattr(self.hooks, "on_tool_result"):
            tool_result_handler = self.create_wrapper_with_context(self.hooks.on_tool_result)
        else:
            tool_result_handler = None
        
        return PocketAgentClient(mcp_config=mcp_config,
                     on_tool_error=self.create_wrapper_with_context(self.hooks.on_tool_error),
                     tool_result_handler=tool_result_handler,
                     **client_kwargs)

    def create_wrapper_with_context(self, func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            hook_context = self._create_hook_context()
            return await func(hook_context, *args, **kwargs)
        return wrapper

    def _format_messages(self) -> list[dict]:
        # format system prompt and messages in proper format
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.messages
        ]
        self.logger.debug(f"Formatted messages: {messages}")
        return messages



    async def _get_llm_response(self, **override_completion_kwargs) -> LitellmModelResponse:
        self.logger.debug(f"Requesting LLM response with model={self.agent_config.llm_model}, message_count={len(self.messages) + 1}")
        # get a response from the llm
        kwargs = self.agent_config.get_completion_kwargs()
        kwargs.update(override_completion_kwargs)
        messages = self._format_messages()
        tools = await self.mcp_client.get_tools(format="openai")
        kwargs.update({
            "tools": tools,
        })
        try:
            self.logger.debug(f"Requesting LLM response with kwargs={kwargs}")
            response = await self.llm_completion_handler.acompletion(
                model=self.agent_config.llm_model,
                messages=messages,
                **kwargs
            )
            
            self.logger.debug(f"LLM response received: full_response={response}")
            return response
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {e}")
            raise
    


    async def add_message(self, message: dict) -> None:
        self.logger.debug(f"Adding message: {message}")
        event = AgentEvent(event_type="new_message", data=message)
        await self.hooks.on_event(event)
        self.messages.append(message)

    
    async def add_llm_message(self, llm_message: LitellmMessage) -> None:
        await self.add_message(llm_message.model_dump())

    async def add_tool_result_message(self, tool_result_message: dict) -> None:
        await self.add_message(tool_result_message)


    def _filter_images_from_tool_result_content(self, tool_result_content: list[dict]) -> list[dict]:
        return [content for content in tool_result_content if content["type"] != "image_url"]



    async def _call_single_tool_with_hooks(self, tool_call: MCPCallToolRequestParams) -> ToolResult:
        """Execute a single tool call with hooks."""
        hook_context = self._create_hook_context()
    
        transformed_tool_call = await self.hooks.pre_tool_call(hook_context, tool_call)
        if transformed_tool_call is not None:
            tool_call = transformed_tool_call
        try:
            result = await self.mcp_client.call_tool(tool_call)
            transformed_result = await self.hooks.post_tool_call(hook_context, tool_call, result)
            if transformed_result is not None:
                result = transformed_result
            return result
        except Exception as e:
            self.logger.error(f"Tool call failed and was not handled by tool error hook: {e}")
            raise


    async def _call_tools(self, tool_calls: list[ChatCompletionMessageToolCall]) -> list[ToolResult]:
        """Execute all tool calls in parallel, with individual hooks for each."""
        transformed_tool_calls = [self.mcp_client.transform_tool_call_request(tool_call) for tool_call in tool_calls]
        tool_results = await asyncio.gather(*[
            self._call_single_tool_with_hooks(tool_call) 
            for tool_call in transformed_tool_calls
        ])
        return tool_results


    async def step(self, **override_completion_kwargs) -> dict:
        self.logger.debug("Starting agent step")
        hook_context = self._create_hook_context()
        await self.hooks.pre_step(hook_context)
        
        step_result = None
        try:
            llm_response = await self._get_llm_response(**override_completion_kwargs)
            await self.hooks.on_llm_response(hook_context, llm_response)
            llm_message = llm_response.choices[0].message
            await self.add_llm_message(llm_message)
            if llm_message.tool_calls:
                tool_names = [
                    tool_call.get('function', {}).get('name', 'unknown') 
                    for tool_call in llm_message.tool_calls
                    ]
                self.logger.debug(f"Executing {len(llm_message.tool_calls)} tool calls: {tool_names}")

                tool_execution_results = await self._call_tools(llm_message.tool_calls)
                if tool_execution_results:
                    self.logger.debug(f"Received tool execution results: {tool_execution_results}")
                    for tool_execution_result in tool_execution_results:
                        tool_call_id = tool_execution_result.tool_call_id
                        tool_call_name = tool_execution_result.tool_call_name
                        tool_result_content = tool_execution_result.tool_result_content
                        if not self.allow_images:
                            self.logger.debug("allow images set to false, filtering images from tool result content")
                            tool_result_content = self._filter_images_from_tool_result_content(tool_result_content)
                        new_message = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_call_name,
                            "content": tool_result_content
                        }
                        await self.add_tool_result_message(new_message)
                        step_result = StepResult(llm_message=llm_message, tool_execution_results=tool_execution_results)
                else:
                    self.logger.error("No tool execution results received")
                    raise ValueError("No tool execution results received. Tool calls must have failed silently.")
            else:
                self.logger.debug("No tool calls in generate result")
                step_result = StepResult(llm_message=llm_message)
        except Exception as e:
            self.logger.error(f"Error in step: {traceback.format_exc()}")
            raise
        finally:
            # Post-step hook
            await self.hooks.post_step(hook_context)
            
            if step_result is None:
                self.logger.debug("Step result is None")
            return step_result

    

    async def add_user_message(self, user_message: str, image_base64s: Optional[list[str]] = None) -> None:
        image_count = len(image_base64s) if image_base64s else 0
        self.logger.info(f"Adding user message: {user_message} with {image_count} images")
        new_message_content = [
            {
                "type": "text",
                "text": user_message
            }
        ]
        if not self.allow_images:
            if image_base64s:
                self.logger.warning("allow images set to false, but images were provided, ignoring images")
            image_base64s = None
            
        else:
            if image_base64s:
                for image_base64 in image_base64s:
                    new_message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    })
        await self.add_message({
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

    def _create_hook_context(self) -> HookContext:
        """Create a hook context for the current state"""
        return HookContext(
            agent=self,
            metadata={}
        )


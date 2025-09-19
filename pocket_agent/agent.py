from litellm import Router
import litellm
from litellm.types.utils import (
    ChatCompletionMessageToolCall, 
    Message as LitellmMessage, 
    ModelResponse as LitellmModelResponse
)
from mcp.types import (
    CallToolRequestParams as MCPCallToolRequestParams,
    TextContent as MCPTextContent,
    ImageContent as MCPImageContent,
)
from abc import abstractmethod
from fastmcp.client.logging import LogMessage
from fastmcp.tools.tool import ToolResult as FastMCPToolResult, Tool as FastMCPTool
from fastmcp import FastMCP
from pydantic import Field
import uuid
import asyncio  # Add this import
from typing import Optional, Tuple

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Union
import json
import traceback


from pocket_agent.client import PocketAgentClient, PocketAgentToolResult
from pocket_agent.utils.logger import configure_logger as configure_pocket_agent_logger


@dataclass
class AgentEvent:
    event_type: str
    data: dict
    meta: Optional[dict] = field(default_factory=dict)

@dataclass
class StepResult:
    llm_message: LitellmMessage
    tool_execution_results: Optional[list[PocketAgentToolResult]] = None


@dataclass
class AgentConfig:
    """Configuration class to make agent setup cleaner"""
    llm_model: str
    name: Optional[str] = None
    role_description: Optional[str] = None
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
    
    async def post_tool_call(self, context: HookContext, tool_call: MCPCallToolRequestParams, result: PocketAgentToolResult) -> Optional[PocketAgentToolResult]:
        return result  # Return modified or None to indicate error

    async def on_llm_response(self, context: HookContext, response: LitellmModelResponse) -> None:
        pass
    
    async def on_event(self, event: AgentEvent) -> None:
        from pocket_agent.utils.console_formatter import ConsoleFormatter
        formatter = ConsoleFormatter()
        formatter.format_event(event)

    #########################################################
    # The following 2 hooks run in the PocketAgentClient
    #########################################################

    # tool result handler will replace the default tool result handler in PocketAgentClient if implemented
    # async def on_tool_result(self, context: HookContext, tool_call: ChatCompletionMessageToolCall, tool_result: FastMCPCallToolResult) -> PocketAgentToolResult:
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
                 mcp_config: Optional[Union[dict, FastMCP]] = None,
                 router: Optional[Router] = None,
                 logger: Optional[logging.Logger] = None,
                 hooks: Optional[AgentHooks] = None,
                 sub_agents: Optional[list["PocketAgent"]] = None,
                 **client_kwargs):

        self.logger = logger or configure_pocket_agent_logger()
        self.agent_config = agent_config

        if not self.agent_config.agent_id:
            self.agent_config.agent_id = str(uuid.uuid4())

        if not agent_config.name:
            self.logger.warning("Agent name is not set, using agent_id as name")
            self.agent_config.name = self.agent_config.agent_id

        if router:
            self.llm_completion_handler = router
        else:
            self.llm_completion_handler = litellm
        self.agent_config = agent_config
        self.hooks = hooks or AgentHooks()

        self._run_lock = asyncio.Lock()

        if not mcp_config and not sub_agents:
            error_message = "MCP config is empty and no sub agents are provided. At least one of the two must be provided."
            self.logger.error(error_message)
            raise ValueError(error_message)
        
        self.mcp_config = mcp_config
        self.sub_agents = sub_agents

        # Mark sub agents as sub agents
        if self.sub_agents:
            for sub_agent in self.sub_agents:
                sub_agent.is_sub_agent = True
            
    
        self.system_prompt = agent_config.system_prompt or ""
        self.messages = agent_config.messages or []

        self.logger.info(
            f"Initializing PocketAgent '{self.name}'",
            extra={
                'agent_id': self.agent_id,
                'model': self.agent_config.llm_model,
                'has_system_prompt': bool(self.system_prompt),
                'initial_messages': len(self.messages),
                'allow_images': self.allow_images,
                'has_sub_agents': self.has_sub_agents,
                'sub_agent_count': self.sub_agent_count
            }
        )

        self._init_client(self.mcp_config, self.sub_agents, **client_kwargs)
        
        # After successful initialization:
        self.logger.info(
            f"PocketAgent '{self.name}' ready - {len(self.messages)} messages, "
            f"{'with' if self.has_sub_agents else 'no'} sub-agents"
        )


    #########################################################
    # Properties
    #########################################################
    @property
    def model(self) -> str:
        return self.agent_config.llm_model

    @property
    def name(self) -> str:
        """Agent name"""
        return self.agent_config.name

    @property  
    def agent_id(self) -> str:
        """Unique agent identifier"""
        return self.agent_config.agent_id

    @property
    def role_description(self) -> Optional[str]:
        """Description of the agent's role/purpose"""
        return self.agent_config.role_description

    @property
    def completion_kwargs(self) -> Dict[str, Any]:
        """LLM completion parameters"""
        return self.agent_config.get_completion_kwargs()

    @property
    def message_count(self) -> int:
        """Number of messages in conversation history"""
        return len(self.messages)

    @property
    def has_sub_agents(self) -> bool:
        """Whether this agent has sub-agents"""
        if self.sub_agents is None:
            return False
        return len(self.sub_agents) > 0

    @property
    def sub_agent_count(self) -> int:
        """Number of sub-agents"""
        if self.sub_agents is None:
            return 0
        return len(self.sub_agents)

    @property
    def is_sub_agent(self) -> bool:
        """Whether this agent is a sub-agent"""
        return getattr(self, '_is_sub_agent', False)
    
    @is_sub_agent.setter
    def is_sub_agent(self, value: bool) -> None:
        self._is_sub_agent = value

    @property
    def is_configured(self) -> bool:
        """Whether the agent is properly configured"""
        return bool(self.agent_config.llm_model and self.mcp_client)

    @property
    def allow_images(self) -> bool:
        return self.agent_config.allow_images



    def _init_client(self, mcp_config: Union[dict, FastMCP, None], sub_agents: Optional[list["PocketAgent"]] = None, **client_kwargs) -> None:
        """
        Initialize the most basic MCP client with the given configuration.
        Override this to add custom client handlers.
        """
        # check if on_tool_result is in hooks class
        if hasattr(self.hooks, "on_tool_result"):
            tool_result_handler = self.create_wrapper_with_context(self.hooks.on_tool_result)
        else:
            tool_result_handler = None

        if isinstance(mcp_config, dict):
            self.mcp_client = PocketAgentClient(mcp_config=mcp_config,
                        on_tool_error=self.create_wrapper_with_context(self.hooks.on_tool_error),
                        tool_result_handler=tool_result_handler,
                        **client_kwargs)
            if sub_agents:
                composed_sub_agents_server = self._create_composed_sub_agents_server()
                self._mount_server_to_mcp_client(composed_sub_agents_server)
        elif isinstance(mcp_config, FastMCP):
            if sub_agents:
                composed_sub_agents_server = self._create_composed_sub_agents_server()
                mcp_config.mount(composed_sub_agents_server)
            self.mcp_client = PocketAgentClient(mcp_config=mcp_config,
                        on_tool_error=self.create_wrapper_with_context(self.hooks.on_tool_error),
                        tool_result_handler=tool_result_handler,
                        **client_kwargs)
        elif sub_agents:
            composed_sub_agents_server = self._create_composed_sub_agents_server()
            self.mcp_client = PocketAgentClient(mcp_config=composed_sub_agents_server,
                        on_tool_error=self.create_wrapper_with_context(self.hooks.on_tool_error),
                        tool_result_handler=tool_result_handler,
                        **client_kwargs)
        else:
            raise ValueError(f"Failed to initialize mcp client: expected mcp config to be of type \
                dict, FastMCP, or None, but got {type(mcp_config)} and/or sub agents to be of type list, but got {type(sub_agents)}.\
                both cannot be None or invalid types.")


                        


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
            self.logger.error(
                f"LLM request failed - Model: {self.agent_config.llm_model}, "
                f"Messages: {len(messages)}, Tools: {len(tools) if tools else 0}",
                exc_info=True,
                extra={
                    'agent_id': self.agent_id,
                    'model': self.agent_config.llm_model,
                    'message_count': len(messages)
                }
            )
            raise
    


    async def add_message(self, message: dict, **meta) -> None:
        self.logger.debug(f"Adding message: {message}")
        meta.update({"agent_name": self.agent_config.name, "is_sub_agent": self.is_sub_agent})
        event = AgentEvent(event_type="new_message", data=message, meta=meta)
        await self.hooks.on_event(event)
        self.messages.append(message)

    
    async def add_llm_message(self, llm_message: LitellmMessage) -> None:
        await self.add_message(llm_message.model_dump())

    async def add_tool_result_message(self, tool_result_message: dict) -> None:
        await self.add_message(tool_result_message)


    def _filter_images_from_tool_result_content(self, tool_result_content: list[dict]) -> list[dict]:
        return [content for content in tool_result_content if content["type"] != "image_url"]



    async def _call_single_tool_with_hooks(self, tool_call: MCPCallToolRequestParams) -> PocketAgentToolResult:
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
            self.logger.error(
                f"Tool call '{tool_call.name}' failed and was not handled by tool error hook",
                exc_info=True,
                extra={
                    'agent_id': self.agent_id,
                    'tool_name': tool_call.name,
                    'tool_arguments': str(tool_call.arguments)[:200],  # Limit length
                    'error_type': type(e).__name__
                }
            )
            raise


    async def _call_tools(self, tool_calls: list[ChatCompletionMessageToolCall]) -> list[PocketAgentToolResult]:
        """Execute all tool calls in parallel, with individual hooks for each."""
        transformed_tool_calls = [self.mcp_client.transform_tool_call_request(tool_call) for tool_call in tool_calls]
        tool_results = await asyncio.gather(*[
            self._call_single_tool_with_hooks(tool_call) 
            for tool_call in transformed_tool_calls
        ])
        return tool_results


    async def step(self, **override_completion_kwargs) -> dict:
        """
        Execute a single step of the agent.
        A step includes a single LLM response and execution of any tool calls that were found in the LLM response.
        """
        self.logger.debug("Starting agent step")
        hook_context = self._create_hook_context()
        await self.hooks.pre_step(hook_context)
        
        step_result = None
        step_phase = "llm_response"
        try:
            llm_response = await self._get_llm_response(**override_completion_kwargs)
            await self.hooks.on_llm_response(hook_context, llm_response)
            llm_message = llm_response.choices[0].message
            await self.add_llm_message(llm_message)
            if llm_message.tool_calls:
                step_phase = "tool_execution"
                tool_details = [
                    f"{tool_call.function.name}({len(tool_call.function.arguments)} args)"
                    for tool_call in llm_message.tool_calls
                ]
                self.logger.info(
                    f"Executing {len(llm_message.tool_calls)} tool calls: {', '.join(tool_details)}",
                    extra={
                        'agent_id': self.agent_id,
                        'tool_count': len(llm_message.tool_calls),
                        'tool_details': tool_details
                    }
                )
                tool_execution_results = await self._call_tools(llm_message.tool_calls)
                if tool_execution_results:
                    step_phase = "tool_execution_result_processing"
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
            self.logger.error(f"Error in step: {step_phase} phase", exc_info=True, extra={
                'agent_id': self.agent_id,
                'step_phase': step_phase,
                'message_count': len(self.messages),
                'has_tool_calls': 'llm_message' in locals() and bool(llm_message.tool_calls)
            })
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


    async def _run_with_lock(self, *args, **kwargs) -> Union[FastMCPToolResult, dict, str]:
        """
        Thread-safe wrapper around run() method
        """
        async with self._run_lock:
            self.logger.debug(f"Acquired run lock for agent {self.name}")
            try:
                return await self.run(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in run: {traceback.format_exc()}")
                raise
            finally:
                self.logger.debug(f"Released run lock for agent {self.name}")


    @abstractmethod
    async def run(self, *args, **kwargs) -> Union[FastMCPToolResult, dict, str]:
        """
        Run the agent.
        Returns the the final result as a FastMCPToolResult, dict, or str.
        """
        pass


    def _create_hook_context(self) -> HookContext:
        """Create a hook context for the current state"""
        return HookContext(
            agent=self,
            metadata={}
        )


    def _format_composed_sub_agent_server_name(self) -> str:
        """Format the composed sub agent server name"""
        return f"interact_with_agent"

    def _format_sub_agent_tool_name(self, agent_name: str) -> str:
        """Format the sub agent tool name"""
        return f"{agent_name}-message"

    def _format_sub_agent_server_description(self, agent_name: str, agent_role_description: str) -> str:
        """Format the sub agents server description"""
        agent_tool_description = f"Send a message to the {agent_name} agent."
        if agent_role_description:
            agent_tool_description += f"Purpose of the {agent_name} agent: \n{agent_role_description}"
        else:
            self.logger.warning(f"Agent role description is not set, sub agent tool description only contains ({agent_tool_description})")
        return agent_tool_description


    def _mount_server_to_mcp_client(self, server: FastMCP) -> None:
        self.mcp_client.mount_server(server)


    def _create_composed_sub_agents_server(self) -> FastMCP:
        """Create a composed sub agents server"""
        server_name = self._format_composed_sub_agent_server_name()
        composed_sub_agents_server = FastMCP(name=server_name)
        for sub_agent in self.sub_agents:
            sub_agent_tool = sub_agent.as_mcp_tool()
            composed_sub_agents_server.add_tool(sub_agent_tool)
        return composed_sub_agents_server


    def as_mcp_tool(self) -> FastMCPTool:
        """
        Return the agent as an in-memory sub-agent server.
        """
        # get agent name and role description
        agent_name = self.agent_config.name
        agent_role_description = self.agent_config.role_description
        agent_tool_name = self._format_sub_agent_tool_name(agent_name)
        agent_tool_description = self._format_sub_agent_server_description(agent_name, agent_role_description)

        async def message_agent(message: str = Field(..., description=f"The message to send to the {agent_name} agent")) -> FastMCPToolResult:
            # run the agent
            result = await self._run_with_lock(message)
            if isinstance(result, FastMCPToolResult):
                return result
            elif isinstance(result, dict):
                return FastMCPToolResult(
                    content=[
                        MCPTextContent(
                            type="text",
                            text=json.dumps(result),
                            meta={"mime_type": "application/json"}
                        )
                    ]
                )
            elif isinstance(result, str):
                return FastMCPToolResult(
                    content=[
                        MCPTextContent(
                            type="text",
                            text=result,
                            meta={"mime_type": "text/plain"}
                        )
                    ]
                )
            elif result is None:
                return FastMCPToolResult(
                    content=[
                        MCPTextContent(
                            type="text",
                            text="No result returned",
                            meta={"mime_type": "text/plain"}
                        )
                    ]
                )
            else:
                raise ValueError(f"Unsupported result type: {type(result)}. Result must be a FastMCPToolResult, dict, or str.")

        return FastMCPTool.from_function(
            fn=message_agent,
            name=agent_tool_name,
            description=agent_tool_description
        )

    def as_mcp_server(self) -> FastMCP:
        """
        Return the agent as its own MCP server.
        """
        mcp_server = FastMCP(name=self.agent_config.name, instructions=self.agent_config.role_description)
        mcp_server.add_tool(self.as_mcp_tool())
        return mcp_server



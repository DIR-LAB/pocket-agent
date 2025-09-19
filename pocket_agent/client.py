from fastmcp import Client as MCPClient, FastMCP as FastMCPServer
from fastmcp.client.client import CallToolResult as FastMCPCallToolResult
from fastmcp.exceptions import ToolError
from fastmcp.client.logging import LogMessage
from mcp.types import (
    ListToolsResult, 
    ListResourcesResult, 
    ListResourceTemplatesResult, 
    CallToolResult as MCPCallToolResult,
    CallToolRequestParams as MCPCallToolRequestParams
)
from litellm.experimental_mcp_client.tools import (
                transform_mcp_tool_to_openai_tool,
                transform_openai_tool_call_request_to_mcp_tool_call_request,
            )
from litellm.types.utils import ChatCompletionMessageToolCall
from dataclasses import dataclass
from typing import Optional, Callable, Literal, Union
import asyncio
import logging
import copy



@dataclass
class PocketAgentToolResult:
    tool_call_id: str
    tool_call_name: str
    tool_result_content: list[dict]
    _extra: Optional[dict] = None


class PocketAgentClient:
    def __init__(self, mcp_config: Union[dict, FastMCPServer], 
                 mcp_logger: Optional[logging.Logger] = None,
                 log_handler: Optional[Callable[[LogMessage], None]] = None,
                 client_logger: Optional[logging.Logger] = None,
                 on_tool_error: Optional[Callable[[ChatCompletionMessageToolCall, Exception], bool]] = None,
                 mcp_server_query_params: Optional[dict] = None,
                 tool_result_handler: Optional[Callable[[ChatCompletionMessageToolCall, FastMCPCallToolResult], PocketAgentToolResult]] = None,
                 **kwargs
                 ):

        # Separate loggers for different purposes
        self.client_logger = client_logger or logging.getLogger("pocket_agent.client")
        self.mcp_logger = mcp_logger or logging.getLogger("pocket_agent.mcp")
        self.mcp_log_handler = log_handler or self._default_mcp_log_handler

        if isinstance(mcp_config, FastMCPServer):
            self.client_logger.info(
                f"Using FastMCP server transport: {mcp_config.name}",
                extra={
                    'transport_type': 'fastmcp_server',
                    'server_name': mcp_config.name
                }
            )
            self.mcp_server_config = mcp_config
        else:
            self.client_logger.info(
                f"Using MCP config as transport: {mcp_config}",
                extra={
                    'transport_type': 'mcp_config',
                    'mcp_config': mcp_config
                }
            )
            self.mcp_server_config = copy.deepcopy(mcp_config)
            if mcp_server_query_params:
                self.client_logger.info(f"Adding MCP server query params to MCP config")
                self.mcp_server_config = self._add_mcp_server_query_params(mcp_server_query_params)

        # Store kwargs for client recreation
        self.client_init_kwargs = kwargs
        
        # Pass MCP log handler to underlying MCP client using factory method
        self.client = self._create_mcp_client(self.mcp_server_config)
        
        # Store on_tool_error and tool_result_handler
        self.on_tool_error = on_tool_error
        self.tool_result_handler = tool_result_handler or self._default_tool_result_handler


    def _create_mcp_client(self, transport) -> MCPClient:
        """Factory method for creating MCPClient instances with consistent config"""
        return MCPClient(
            transport=transport,
            log_handler=self.mcp_log_handler,
            **self.client_init_kwargs
        )

    # This function allows agents to metadata via query params to MCP servers (e.g. supply a user id) 
    # Using this approach is only temporary until the official MCP Python SDK supports metadata in tool calls
    def _add_mcp_server_query_params(self, mcp_server_query_params: dict) -> dict:
        mcp_config = copy.deepcopy(self.mcp_server_config)
        mcp_servers = mcp_config["mcpServers"]
        for server_name, server_config in mcp_servers.items():
            if "url" in server_config:
                mcp_server_url = server_config["url"]
                for idx, (param, value) in enumerate(mcp_server_query_params.items()):
                    if idx == 0:
                        mcp_server_url += f"?{param}={value}"
                    else:
                        mcp_server_url += f"&{param}={value}"
                mcp_config["mcpServers"][server_name]["url"] = mcp_server_url
            else:
                self.client_logger.warning(f"MCP server {server_name} is not an http server, so query params are not supported")
        return mcp_config



    def _default_mcp_log_handler(self, message: LogMessage):
        """Handle MCP server logs using dedicated MCP logger"""
        LOGGING_LEVEL_MAP = logging.getLevelNamesMapping()
        msg = message.data.get('msg')
        extra = message.data.get('extra', {})
        extra.update({
            'source': 'mcp_server'
        })

        level = LOGGING_LEVEL_MAP.get(message.level.upper(), logging.INFO)
        self.mcp_logger.log(level, f"[MCP] {msg}", extra=extra)


    async def get_tools(self, format: Literal["mcp", "openai"] = "mcp") -> ListToolsResult:
        self.client_logger.debug(f"Getting MCP tools in {format} format")
        async with self.client:
            tools = await self.client.list_tools()
        if format == "mcp":
            self.client_logger.debug(f"MCP tools: {tools}")
            return tools
        elif format == "openai":
            self.client_logger.debug(f"Converting MCP tools to OpenAI format")
            openai_tools = [transform_mcp_tool_to_openai_tool(tool) for tool in tools]
            self.client_logger.debug(f"OpenAI tools: {openai_tools}")
            return openai_tools
        else:
            raise ValueError(f"Invalid tool list format. Expected 'mcp' or 'openai', got {format}")


    async def get_tool_input_format(self, tool_name: str):
        tools = await self.get_tools(format="mcp")
        for tool in tools:
            if tool.name == tool_name:
                return tool.inputSchema
        raise ValueError(f"Tool {tool_name} not found")

    
    def transform_tool_call_request(self, tool_call: ChatCompletionMessageToolCall) -> MCPCallToolRequestParams:
        self.client_logger.debug(f"Transforming tool call request to MCP format: {tool_call}")
        transformed_tool_call = transform_openai_tool_call_request_to_mcp_tool_call_request(openai_tool=tool_call.model_dump())
        transformed_tool_call.id = tool_call.id
        self.client_logger.debug(f"Transformed tool call request to MCP format: {transformed_tool_call}")
        return transformed_tool_call
            

    async def call_tool(self, tool_call: MCPCallToolRequestParams) -> PocketAgentToolResult:
        tool_call_id = tool_call.id
        tool_call_name = tool_call.name
        tool_call_arguments = tool_call.arguments

        try:
            async with self.client:
                self.client_logger.debug(f"Calling tool: {tool_call_name} with arguments: {tool_call_arguments}")
                tool_result = await self.client.call_tool(tool_call_name, tool_call_arguments)
        except ToolError as e:
            # handle tool error
            if self.on_tool_error:
                on_tool_error_result = await self.on_tool_error(tool_call, e)
                if type(on_tool_error_result) == str:
                    tool_result_content = [{
                        "type": "text",
                        "text": on_tool_error_result
                    }]
                    return PocketAgentToolResult(
                        tool_call_id=tool_call_id,
                        tool_call_name=tool_call_name,
                        tool_result_content=tool_result_content
                    )

                # False means the tool error handler did not handle the error
                elif on_tool_error_result == False:
                    self.client_logger.error(
                        f"Tool error handler returned False, which means the tool call should be skipped: '{tool_call_name}'",
                        extra={
                            'tool_name': tool_call_name,
                            'tool_arguments': str(tool_call.arguments)[:200],
                            'error_type': type(e).__name__,
                            'error_message': str(e)
                        }
                    )
                    raise e

                # If we get here, the tool error handler returned an invalid type
                else:
                    error_message = f"Tool error handler returned invalid type: expected str|False, got {type(on_tool_error_result)}"
                    self.client_logger.error(
                        error_message,
                        extra={
                            'tool_name': tool_call_name,
                            'handler_result_type': type(on_tool_error_result).__name__,
                            'handler_result': str(on_tool_error_result)[:100]
                        }
                    )
                    raise ValueError(error_message)
            
            # If we get here, there is no tool error handler
            else:
                self.client_logger.error(
                    f"No tool error handler is set",
                    extra={
                        'tool_name': tool_call_name,
                        'tool_arguments': str(tool_call.arguments)[:200],
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    }
                )
                raise e
        else:
            return self.tool_result_handler(tool_call, tool_result)

    
    def _default_tool_result_handler(self, tool_call: ChatCompletionMessageToolCall, tool_result: FastMCPCallToolResult) -> PocketAgentToolResult:
        """
        The function transforms the fastmcp tool result to a tool result that can be used by the agent.
        The default implementation just extracts text and image content and transforms them into a format that can directly be used as a message
        """
        tool_result_content = []
        tool_result_raw_content = []
        for content in tool_result.content:
            tool_result_raw_content.append(content.model_dump())
            if content.type == "text":
                tool_result_content.append({
                    "type": "text",
                    "text": content.text
                })
            elif content.type == "image":
                tool_result_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content.mimeType};base64,{content.data}"
                    }
                })
        return PocketAgentToolResult(
            tool_call_id=tool_call.id,
            tool_call_name=tool_call.name,
            tool_result_content=tool_result_content,
            _extra={
                "tool_result_raw_content": tool_result_raw_content
            }
        )

    
    def mount_server(self, server: FastMCPServer):
        proxy_server = FastMCPServer.as_proxy(
            self.client
        )
        proxy_server.mount(server)
        self.client = self._create_mcp_client(proxy_server)




        

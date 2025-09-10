from fastmcp import Client as MCPClient
from fastmcp.exceptions import ToolError
from fastmcp.client.logging import LogMessage
from mcp.types import ListToolsResult, CallToolResult as MCPCallToolResult
from litellm.experimental_mcp_client.tools import (
                transform_mcp_tool_to_openai_tool,
                transform_openai_tool_call_request_to_mcp_tool_call_request,
            )
from litellm.types.utils import ChatCompletionMessageToolCall
from dataclasses import dataclass
from typing import Optional, Callable
import asyncio
import logging
import copy



@dataclass
class ToolResult:
    tool_call_id: str
    tool_call_name: str
    tool_result_content: list[dict]


class Client:
    def __init__(self, mcp_server_config: dict, 
                 mcp_logger: Optional[logging.Logger] = None,
                 mcp_log_handler: Optional[Callable[[LogMessage], None]] = None,
                 client_logger: Optional[logging.Logger] = None,
                 elicitation_handler: Optional[Callable[[str], None]] = None,
                 progress_handler: Optional[Callable[[str], None]] = None,
                 sampling_handler: Optional[Callable[[str], None]] = None,
                 message_handler: Optional[Callable[[str], None]] = None,
                 on_tool_error: Optional[Callable[[Exception], bool]] = None,
                 mcp_server_query_params: Optional[dict] = None
                 ):

        self.mcp_server_config = copy.deepcopy(mcp_server_config)
        if mcp_server_query_params:
            server_config_with_params = self._add_mcp_server_query_params(mcp_server_query_params)
            self.mcp_server_config = server_config_with_params

        # Separate loggers for different purposes
        self.client_logger = client_logger or logging.getLogger("pocket_agent.client")
        self.mcp_logger = mcp_logger or logging.getLogger("pocket_agent.mcp")
        self.mcp_log_handler = mcp_log_handler or self._default_mcp_log_handler
        
        
        # Pass MCP log handler to underlying MCP client
        self.client = MCPClient(self.mcp_server_config, 
                                log_handler=mcp_log_handler,
                                elicitation_handler=elicitation_handler, 
                                progress_handler=progress_handler, 
                                sampling_handler=sampling_handler, 
                                message_handler=message_handler
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


    async def _get_mcp_tools(self) -> ListToolsResult:
        async with self.client:
            tools = await self.client.get_tools()
            return tools
    

    async def _get_openai_tools(self) -> list[dict]:
        tools = await self._get_mcp_tools()
        openai_tools = []
        for tool in tools:
            openai_tools.append(transform_mcp_tool_to_openai_tool(tool))
        return openai_tools

    
    async def call_tools(self, tool_calls: list[ChatCompletionMessageToolCall]) -> list[ToolResult]:
        # call tools in parallel
        async with self.client:
            tool_results = await asyncio.gather(*[self.call_tool(tool_call) for tool_call in tool_calls])
            return tool_results
            

    async def _get_tool_format(self, tool_name: str):
        tools = await self._get_mcp_tools()
        for tool in tools:
            if tool.name == tool_name:
                return tool.inputSchema
        raise ValueError(f"Tool {tool_name} not found")
            

    async def call_tool(self, tool_call: ChatCompletionMessageToolCall) -> ToolResult:
        tool_call_id = tool_call.id
        tool_call_name = tool_call.name
        mcp_tool_call_request = transform_openai_tool_call_request_to_mcp_tool_call_request(openai_tool=tool_call.model_dump())
        tool_call_arguments = mcp_tool_call_request.arguments

        try:
            tool_result = await self.client.call_tool(tool_call_name, tool_call_arguments)
        except ToolError as e:
            # if the llm gives a false argument, let it know the expected format
            if "unexpected_keyword_argument" in str(e):
                tool_format = await self._get_tool_format(tool_call_name)
                tool_result_content = [{
                    "type": "text",
                    "text": "You supplied an unexpected keyword argument to the tool. Try again with the correct arguments as specified in expected format: \n" + tool_format
                }]
            else:
                # handle tool error
                if self.on_tool_error:
                    message = self.on_tool_error(e)
                    if type(message) == str:
                        tool_result_content = [{
                            "type": "text",
                            "text": message
                        }]
                    else:
                        raise e
                else:
                    raise e
        else:
            tool_result_content = self._parse_tool_result(tool_result)
        
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_call_name=tool_call_name,
            tool_result_content=tool_result_content)

        
    def _parse_tool_result(self, tool_result: MCPCallToolResult) -> list[dict]:
        if tool_result.structuredContent:
            tool_result_content = [{
                "type": "text",
                "text": tool_result.structuredContent
            }]
            return tool_result_content
        else:
            tool_result_content = []
            for content in tool_result.content:
                if content.type == "text":
                    tool_result_content.append({
                        "type": "text",
                        "text": content.text
                    })
                elif content.type == "image":
                    tool_result_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{content.imageBase64}"
                        }
                    })
            return tool_result_content



        

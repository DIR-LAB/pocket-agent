# PocketAgentClient

Each PocketAgent instance initializes a PocketAgentClient which acts as a wrapper for the [FastMCP Client](https://gofastmcp.com/clients/client) to implement the standard mcp protocol features and some additional features.

## Custom Metadata

- Sending custom metadata is now supported by the MCP protocol. As such, Pocket Agent also supports sending custom metadata to MCP servers during tool calls. By default, Pocket Agent sends the following metadata on each tool call:
    ```python
    {
        "tool_call_id": tool_call.id,
        "context_id": self.agent_id,
        "agent_name": self.agent_config.name,
        "is_sub_agent": self.is_sub_agent
    }
    ```
- To add additional custom values to the metadata, the `PocketAgent` class exposes a  `custom_tool_call_metadata` parameter which accepts a `dict`. Custom metadata will be merged with the default metadata. For example:
    ```python
    # Create a PocketAgent instance with custom metadata
    agent = PocketAgent(
        ...
        custom_tool_call_metadata = {
            "context_id": "1111",
            "agent_category": "researcher"
        }
    )

    # Every tool call from the agent will have its `meta` set to this:
    {
        "tool_call_id": tool_call.id,
        "context_id": "1111",
        "agent_name": self.agent_config.name,
        "is_sub_agent": self.is_sub_agent,
        "agent_category": "researcher"
    }
    ```
- In order to access tool call metadata from within an MCP tool you can simply use the automatically injected Context:
    ```python
    BASE_PATHS = {
        "1432": "home/dir/1432"
        "1111": "home/dir/1111"
    }
    @mcp.tool
    def save_file(file_name: str, file_content: str, ctx: Context) -> str:
        meta = ctx.request_context.meta
        base_path = "/dir/0"
        if meta:
            context_id = meta.context_id if hasattr(meta, 'context_id') else None
            if context_id:
                base_path = BASE_PATHS.get(context_id, base_path)
        file_path = os.path.join(base_path, file_name)
        # Save file to path

        return f"File saved"
    ```

## on_tool_error (hook)

- If a tool call fails and is not handled within the tool itself, it will result in a ToolError result. You can add custom handling of such errors using the `on_tool_error` hook method in `AgentHooks`. Any custom functionality should either return a `string` or `False`. If the method returns a `string`, the contents will be sent to the agent as the tool result, if the method returns `False` the ToolError will be raised an execution of the agent will stop.
The following handler is implemented by default to handle a common scenario where LLMs pass invalid parameters to tools resulting in an error:

    ```python
    class AgentHooks:
        async def on_tool_error(self, context: HookContext, tool_call: MCPCallToolRequestParams, error: Exception) -> Union[str, False]:
            if "unexpected_keyword_argument" in str(error):
                tool_call_name = tool_call.name
                tool_format = await context.agent.mcp_client.get_tool_input_format(tool_call_name)
                return "You supplied an unexpected keyword argument to the tool. \
                    Try again with the correct arguments as specified in expected format: \n" + tool_format
            return False
    ```

## tool_result_handler (hook)

- When a tool is called successfully it results in a [CallToolResult](https://github.com/jlowin/fastmcp/blob/09ae8f5cfdc62e6ac0bb7e6cc8ade621e9ebbf3e/src/fastmcp/client/client.py#L935). Most of the time, you will likely just want to parse the content which is a list of MCP content objects (i.e. [TextContent, ImageContent, etc](https://github.com/modelcontextprotocol/python-sdk/blob/c3717e7ad333a234c11d084c047804999a88706e/src/mcp/types.py#L662)). For this reason, the `PocketAgentClient` uses its default parser to parse these objects into content that can directly be fed to the agent as a message. Specifically, the default parser will return a ToolResult object:

    ```python
    return ToolResult(
        tool_call_id=tool_call.id,                              # ID of the original tool call (needed by most apis when passing tool results)
        tool_call_name=tool_call.name,                          # Name of the tool which the result is for
        tool_result_content=tool_result_content,                # Tool result content compatible with LiteLLM message format
        _extra={
            "tool_result_raw_content": tool_result_raw_content  # Unprocessed MCP tool result (unused by default)
        }
    )
    ```

    However, in some cases you may want to specifically parse structured content from a known tool in which case you can override the default parser by implementing the `on_tool_result` hook method in `AgentHooks`:

    ```python
    class CustomHooks(AgentHooks):
        async def on_tool_result(self, context: HookContext, tool_call: ChatCompletionMessageToolCall, tool_result: FastMCPCallToolResult) -> ToolResult:
            # your custom tool result parsing
    ```

## Server-initiated Events
- The MCP protocol implements numerous server-initiated events which should be handled by MCP clients. Each of these are documented here:
     - [Elicitation](https://gofastmcp.com/clients/elicitation)
     - [Logging](https://gofastmcp.com/clients/logging)
     - [Progress](https://gofastmcp.com/clients/progress)
     - [Sampling](https://gofastmcp.com/clients/sampling)
     - [Messages](https://gofastmcp.com/clients/messages)

By default, PocketAgent only implements the logging handler.

To define custom behavior for any other server initiated events they can be passed as additional arguments to the agent:
```python
agent = PocketAgent(
    elicitation_handler=your_elicitation_handler,
    log_handler=your_log_handler,
    progress_handler=your_progress_handler,
    sampling_handler=your_sampling_handler,
    message_handler=your_message_handler
)
```

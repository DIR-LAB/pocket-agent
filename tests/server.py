from fastmcp import FastMCP, Context
import asyncio



"""Create a test FastMCP server with basic tools"""
server = FastMCP(
    name="TestServer",
    instructions="This server provides tools for testing the pocket agent framework"
)


@server.tool(description="Greet someone by name and tell them what the session context_id is")
def greet(name: str, ctx: Context) -> str:
    """Greet someone by name."""
    meta = ctx.request_context.meta
    print(f"Meta: {meta}")
    if meta:
        context_id = meta.context_id if hasattr(meta, 'context_id') else None
        if context_id:
            return f"Hello, {name}. You are in context {context_id}!"
    return f"Hello, {name}!"

@server.tool(description="Add two numbers together")
def add(a: int, b: int, ctx: Context) -> int:
    """Add two numbers together."""
    return a + b

@server.tool(description="Sleep for a given number of seconds")
async def sleep(seconds: float, ctx: Context) -> str:
    """Sleep for a given number of seconds."""
    await asyncio.sleep(seconds)
    return f"Slept for {seconds} seconds"

@server.tool(description="Multiply two numbers")
def multiply(a: int, b: int, ctx: Context) -> int:
    """Multiply two numbers together."""
    return a * b

@server.tool(description="Get current status")
def get_status(ctx: Context) -> str:
    """Get the current server status."""
    return "Server is running and ready"

@server.tool(description="Get the custom tool call metadata")
def get_tool_call_metadata(ctx: Context) -> dict:
    """Get the custom tool call metadata."""
    return ctx.request_context.meta

# Add a resource for testing
@server.resource(uri="data://test", description="Test resource")
async def get_test_data():
    return {"message": "This is test data", "timestamp": "2024-01-01"}



if __name__ == "__main__":
    server.run()
from fastmcp import FastMCP
import random
import time
import logging


"""Create a test FastMCP server with basic tools"""
server = FastMCP(
    name="TestServer",
    instructions="This server provides tools for testing the pocket agent framework"
)

@server.tool(description="Greet someone by name")
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

@server.tool(description="Add two numbers together")
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@server.tool(description="Sleep for a given number of seconds")
async def sleep(seconds: float) -> str:
    """Sleep for a given number of seconds."""
    await asyncio.sleep(seconds)
    return f"Slept for {seconds} seconds"

@server.tool(description="Multiply two numbers")
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@server.tool(description="Get current status")
def get_status() -> str:
    """Get the current server status."""
    return "Server is running and ready"

# Add a resource for testing
@server.resource(uri="data://test", description="Test resource")
async def get_test_data():
    return {"message": "This is test data", "timestamp": "2024-01-01"}



if __name__ == "__main__":
    server.run(show_banner=False)
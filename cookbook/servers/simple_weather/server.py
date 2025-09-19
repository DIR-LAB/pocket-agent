from fastmcp import FastMCP
import random
import time
import logging
import asyncio

# Suppress FastMCP's verbose logging so it doesnt interfere with the agent's console output
logging.getLogger("FastMCP").setLevel(logging.WARNING)

weather_mcp = FastMCP(name="weather",
                     instructions="This server provides tools to get weather information for cities")

# Sample weather data for demo purposes
WEATHER_DATA = {
    "london": "Cloudy with light rain, 15°C",
    "paris": "Sunny and warm, 22°C", 
    "tokyo": "Partly cloudy, 18°C",
    "new york": "Thunderstorms, 16°C",
    "sydney": "Clear skies, 25°C",
    "moscow": "Snow, -5°C",
    "mumbai": "Hot and humid, 32°C",
    "cairo": "Sunny and hot, 35°C"
}

WEATHER_CONDITIONS = [
    "sunny", "cloudy", "rainy", "snowy", "windy", "foggy", 
    "partly cloudy", "thunderstorms", "clear skies", "overcast"
]

@weather_mcp.tool(
    description="Get the current weather for a specific city"
)
async def get_weather(city: str) -> str:
    """Get weather information for a city."""
    await asyncio.sleep(5)
    city_lower = city.lower()
    
    if city_lower in WEATHER_DATA:
        return f"🌤️ Weather in {city.title()}: {WEATHER_DATA[city_lower]}"
    else:
        # Generate random weather for unknown cities
        condition = random.choice(WEATHER_CONDITIONS)
        temp = random.randint(-10, 40)
        return f"🌤️ Weather in {city.title()}: {condition.title()}, {temp}°C"

@weather_mcp.tool(
    description="Get a weather forecast for multiple days"
)
async def get_forecast(city: str, days: int = 3) -> str:
    """Get a multi-day weather forecast for a city."""
    await asyncio.sleep(5)
    if days > 7:
        days = 7  # Limit to 7 days
    
    forecast = f"📅 {days}-day forecast for {city.title()}:\n"
    
    for day in range(1, days + 1):
        condition = random.choice(WEATHER_CONDITIONS)
        temp = random.randint(-5, 35)
        forecast += f"Day {day}: {condition.title()}, {temp}°C\n"
    
    return forecast.strip()

if __name__ == "__main__":
    weather_mcp.run(show_banner=False)



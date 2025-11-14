"""Shared local tools for both realtime and non-realtime agents."""
from __future__ import annotations

import logging
from typing import Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Optional HTTP client for weather
try:
    import httpx
except Exception:
    httpx = None

# Agents SDK tools (for realtime)
try:
    from agents import function_tool
except Exception:
    function_tool = None

logger = logging.getLogger(__name__)


def current_time(timezone_name: Optional[str] = None) -> str:
    """Return the current time in ISO 8601. Optional timezone as an IANA string, e.g., 'America/Toronto'."""
    try:
        if timezone_name:
            return datetime.now(ZoneInfo(timezone_name)).isoformat()
        return datetime.now(timezone.utc).isoformat()
    except Exception as e:
        return f"current_time error: {e}"


async def weather_now(city: str) -> str:
    """Get current weather for a city using Open-Meteo (no API key)."""
    if not httpx:
        return "weather_now unavailable: httpx not installed."
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Geocode
            g = (await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
            )).json()
            if not g.get("results"):
                return f"Couldn't find '{city}'."
            r0 = g["results"][0]
            lat, lon = r0["latitude"], r0["longitude"]
            loc = r0.get("name") or city

            # Current weather
            w = (await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current_weather": True},
            )).json()
            cw = w.get("current_weather") or {}
            t = cw.get("temperature")
            wind = cw.get("windspeed")
            code = cw.get("weathercode")
            ts = cw.get("time")
            return f"Weather in {loc}: {t}°C, wind {wind} km/h, code {code}, at {ts}."
    except Exception as e:
        return f"weather_now failed: {e}"


# Realtime agent tools (decorated with @function_tool)
def get_realtime_tools():
    """Get tools decorated for realtime agent use."""
    if function_tool is None:
        logger.warning("agents SDK not available, realtime tools will not work")
        return []
    
    @function_tool(
        name_override="current_time",
        description_override="Return the current time in ISO 8601. Optional timezone as an IANA string, e.g., 'America/Toronto'.",
    )
    def current_time_tool(timezone_name: Optional[str] = None) -> str:
        return current_time(timezone_name)
    
    @function_tool(
        name_override="weather_now",
        description_override="Get current weather for a city using Open-Meteo (no API key).",
    )
    async def weather_now_tool(city: str) -> str:
        return await weather_now(city)
    
    return [current_time_tool, weather_now_tool]


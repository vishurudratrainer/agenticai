import os
from mcp.server.fastmcp import FastMCP
import requests
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
mcp = FastMCP("LogAnalyzer")

@mcp.tool()
def get_country_info(country_name: str) -> Optional[Dict[str, Any]]:
    """Fetch geographical coordinates and details using Open-Meteo Geocoding API."""
    query = country_name.strip()
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=en&format=json"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results")
            if results and len(results) > 0:
                target = results[0]
                return {
                    "name": target.get("name", country_name),
                    "country": target.get("country", country_name),
                    "country_code": target.get("country_code", "N/A").upper(),
                    "population": target.get("population", 0),
                    "latlng": [target.get("latitude", 0.0), target.get("longitude", 0.0)],
                    "timezone": target.get("timezone", "UTC"),
                }
    except requests.RequestException as e:
        print(f"[ERROR] Geocoding API connection error: {e}")
        
    return None

@mcp.tool()
def get_weather_forecast(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch current weather metrics from Open-Meteo API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current_weather=true"
    )
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current_weather", {})
            return {
                "temperature": current.get("temperature", "N/A"),
                "windspeed": current.get("windspeed", 0),
                "weathercode": current.get("weathercode", 0),
                "time": current.get("time", "N/A"),
            }
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch weather data: {e}")
    return None

@mcp.tool()
def get_nearby_earthquakes(
lat: float, lon: float, max_radius_km: float = 800
) -> List[Dict[str, Any]]:
    """Fetch recent seismic activity from USGS API near given coordinates."""
    url = (
        f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&"
        f"latitude={lat}&longitude={lon}&maxradiuskm={max_radius_km}&limit=5"
    )
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            events = []
            for feat in features:
                props = feat.get("properties", {})
                time_ms = props.get("time", 0)
                formatted_time = (
                    datetime.fromtimestamp(time_ms / 1000).strftime("%Y-%m-%d %H:%M")
                    if time_ms
                    else "Unknown"
                )
                events.append(
                    {
                        "place": props.get("place", "Unknown location"),
                        "mag": props.get("mag", 0.0),
                        "time": formatted_time,
                    }
                )
            return events
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch earthquake data: {e}")
    return []

@mcp.tool()
def calculate_risk_score(
    weather: Dict[str, Any], earthquakes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculates environmental risk score based on windspeed and seismic data."""
    base_score = 100
    penalties = []

    windspeed = weather.get("windspeed", 0) or 0
    if windspeed > 50:
        base_score -= 30
        penalties.append("High Wind Warning (>50 km/h)")
    elif windspeed > 25:
        base_score -= 10
        penalties.append("Moderate Wind Notice")

    eq_count = len(earthquakes)
    valid_mags = [eq["mag"] for eq in earthquakes if eq.get("mag") is not None]
    max_mag = max(valid_mags, default=0.0)

    if max_mag >= 5.0:
        base_score -= 40
        penalties.append(f"Significant Seismic Event (Mag {max_mag})")
    elif eq_count > 0:
        base_score -= 15
        penalties.append(f"Minor Seismic Activity ({eq_count} events nearby)")

    risk_level = "LOW"
    if base_score < 50:
        risk_level = "HIGH"
    elif base_score < 80:
        risk_level = "MODERATE"

    return {
        "score": max(0, base_score),
        "level": risk_level,
        "factors": penalties if penalties else ["No major environmental risks"],
    }


if __name__ == "__main__":
    mcp.run()
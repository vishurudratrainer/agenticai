import asyncio
import json
import logging
import sys

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
# Setup Client Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("MCPBackendClient")

async def execute_mcp_pipeline(target_location: str, ollama_model: str):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.abspath("log_server.py")],
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

async def execute_mcp_pipeline(target_location: str, ollama_model: str):
    try:
        """Executes headless MCP tool calling sequence over stdio transport."""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.abspath("risk_server.py")],
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        logger.info("Starting stdio connection to MCP Server...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("MCP Session initialized successfully.")

                # 1. Discover registered tools on the server
                tools_response = await session.list_tools()
                available_tools = [t.name for t in tools_response.tools]
                logger.info(f"Available Server Tools: {available_tools}")

                # 2. Invoke get_country_info tool
                logger.info(f"Calling 'get_country_info' for '{target_location}'...")
                res_location = await session.call_tool(
                    "get_country_info", arguments={"query_name": target_location}
                )
                location_data = json.loads(res_location.content[0].text)

                if "error" in location_data:
                    logger.error(f"Execution Error: {location_data['error']}")
                    return

                lat = location_data["latitude"]
                lon = location_data["longitude"]

                # 3. Invoke weather and earthquake tools
                logger.info(f"Calling weather and earthquake tools for ({lat}, {lon})...")
                res_weather = await session.call_tool(
                    "get_weather_forecast",
                    arguments={"latitude": lat, "longitude": lon},
                )
                weather_data = json.loads(res_weather.content[0].text)

                res_eq = await session.call_tool(
                    "get_nearby_earthquakes",
                    arguments={"latitude": lat, "longitude": lon},
                )
                earthquake_data = json.loads(res_eq.content[0].text)

                mags = [
                    eq["mag"]
                    for eq in earthquake_data
                    if isinstance(eq, dict) and eq.get("mag") is not None
                ]

                # 4. Invoke calculate_risk_score tool
                logger.info("Calling 'calculate_risk_score'...")
                res_risk = await session.call_tool(
                    "calculate_risk_score",
                    arguments={
                        "windspeed": weather_data.get("windspeed", 0),
                        "earthquake_magnitudes": mags,
                    },
                )
                risk_data = json.loads(res_risk.content[0].text)

                # 5. Synthesize Briefing via Ollama
                prompt = f"""
                You are an expert risk analyst. Synthesize this data retrieved via MCP server tools:
                Location: {location_data['name']} ({location_data['country']})
                Temperature: {weather_data.get('temperature')}°C
                Wind Speed: {weather_data.get('windspeed')} km/h
                Seismic Events: {len(earthquake_data)}
                Safety Score: {risk_data['score']}/100 (Level: {risk_data['level']})
                Risk Factors: {', '.join(risk_data['factors'])}

                Provide a concise, professional 3-bullet executive summary for operational travel readiness.
                """

                logger.info(f"Invoking Ollama model '{ollama_model}' for final synthesis...\n")
                try:
                    response = ollama.chat(
                        model=ollama_model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    
                    # Terminal Payload Output
                    print("\n==================================================")
                    print("            MCP BACKEND PIPELINE OUTPUT           ")
                    print("==================================================")
                    print(f"Target Location : {location_data['name']}, {location_data['country_code']}")
                    print(f"Coordinates     : {lat}, {lon}")
                    print(f"Temperature     : {weather_data.get('temperature')} °C")
                    print(f"Wind Speed      : {weather_data.get('windspeed')} km/h")
                    print(f"Safety Score    : {risk_data['score']}/100 ({risk_data['level']})")
                    print(f"Risk Factors    : {', '.join(risk_data['factors'])}")
                    print("--------------------------------------------------")
                    print("OLLAMA EXECUTIVE BRIEFING:")
                    print(response["message"]["content"])
                    print("==================================================\n")

                except Exception as e:
                    logger.exception(f"Ollama execution failed: {e}")
    except Exception as e:
        print(e)
        logger.exception(f"Failed execution: {e}")





if __name__ == "__main__":
    location = "Japan"
    model =  "mistral"

    asyncio.run(execute_mcp_pipeline(location, model))
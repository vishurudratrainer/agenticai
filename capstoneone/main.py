import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
import requests
import ollama


class OpenAPIClient:
    """Handles communications with reliable, keyless public REST APIs."""

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


class IntelligenceEngine:
    """Processes aggregated API data and generates AI analysis via Ollama."""

    @staticmethod
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

    @staticmethod
    def generate_ai_briefing(
        country: Dict[str, Any],
        weather: Dict[str, Any],
        earthquakes: List[Dict[str, Any]],
        risk: Dict[str, Any],
        model_name: str = "mistral",
    ) -> str:
        """Sends aggregated telemetry to a local Ollama instance for LLM synthesis."""
        prompt = f"""
        You are a senior geopolitical and environmental risk analyst. 
        Synthesize the following real-time telemetry into a concise, professional executive briefing (3-4 bullet points):

        Location Information:
        - Entity/Country: {country['name']} ({country['country']}, Code: {country['country_code']})
        - Population Estimate: {country['population']:,}
        - Timezone: {country['timezone']}

        Telemetry & Metrics:
        - Current Temperature: {weather['temperature']}°C
        - Wind Speed: {weather['windspeed']} km/h
        - Recent Nearby Earthquakes (800km radius): {len(earthquakes)} recorded events
        - Computed Safety Score: {risk['score']}/100 (Risk Level: {risk['level']})
        - Risk Factors: {', '.join(risk['factors'])}

        Focus on operational impacts, travel advisory state, and environmental risks.
        """

        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as e:
            return (
                f"[Ollama Briefing Warning]: Could not reach Ollama model '{model_name}'.\n"
                f"Error Details: {e}\n"
                f"(Make sure 'ollama serve' is running and you have run 'ollama pull {model_name}')"
            )


class ReportPresenter:
    """Formats and prints executive reports to terminal."""

    @staticmethod
    def render_dashboard(
        country: Dict[str, Any],
        weather: Dict[str, Any],
        earthquakes: List[Dict[str, Any]],
        risk: Dict[str, Any],
        ai_summary: str,
    ) -> None:
        print("\n" + "=" * 65)
        print(f"   GLOBAL INTELLIGENCE REPORT: {country['name'].upper()} ({country['country_code']})")
        print("=" * 65)

        print("\n1. GEOGRAPHIC & DEMOGRAPHIC METRICS")
        print(f"   • Location Name : {country['name']}")
        print(f"   • Country       : {country['country']}")
        print(f"   • Population    : {country['population']:,}")
        print(f"   • Coordinates   : {country['latlng'][0]}°, {country['latlng'][1]}°")
        print(f"   • Timezone      : {country['timezone']}")

        print("\n2. REAL-TIME ATMOSPHERIC CONDITIONS")
        print(f"   • Temperature   : {weather['temperature']} °C")
        print(f"   • Wind Speed    : {weather['windspeed']} km/h")
        print(f"   • Timestamp     : {weather['time']}")

        print("\n3. RECENT SEISMIC EVENTS (800 km Radius)")
        if earthquakes:
            for eq in earthquakes:
                print(f"   • Mag {eq['mag']} | {eq['place']} | Date: {eq['time']}")
        else:
            print("   • No recent seismic activity detected.")

        print("\n4. RISK ASSESSMENT ANALYSIS")
        print(f"   • Environmental Safety Score : {risk['score']}/100")
        print(f"   • Risk Level Indicator       : {risk['level']}")
        print("   • Key Factors:")
        for factor in risk["factors"]:
            print(f"     - {factor}")

        print("\n5. OLLAMA AI EXECUTIVE BRIEFING")
        print(ai_summary)

        print("\n" + "=" * 65 + "\n")


def main():
    print("Initializing Global Intelligence Dashboard with Ollama AI...")
    target = input("Enter country or city name (e.g., India, Japan, Peru, Germany): ").strip()

    if not target:
        target = "India"
        print("No input detected. Defaulting to 'India'.")

    # Step 1: Geocoding via Open-Meteo
    country_data = OpenAPIClient.get_country_info(target)
    if not country_data:
        print(f"[ERROR] Could not locate coordinates for '{target}'. Exiting.")
        sys.exit(1)

    lat, lon = country_data["latlng"][0], country_data["latlng"][1]

    # Step 2: Fetch Telemetry
    weather_data = OpenAPIClient.get_weather_forecast(lat, lon)
    earthquake_data = OpenAPIClient.get_nearby_earthquakes(lat, lon)

    if weather_data:
        # Step 3: Compute Rule-based Risk Score
        risk_metrics = IntelligenceEngine.calculate_risk_score(
            weather_data, earthquake_data
        )

        # Step 4: Generate LLM Synthesis via Ollama
        print("\n[INFO] Contacting local Ollama model for AI synthesis...")
        ai_briefing = IntelligenceEngine.generate_ai_briefing(
            country_data, weather_data, earthquake_data, risk_metrics, model_name="llama3"
        )

        # Step 5: Render Dashboard Report
        ReportPresenter.render_dashboard(
            country_data, weather_data, earthquake_data, risk_metrics, ai_briefing
        )
    else:
        print("Failed to aggregate complete intelligence data.")


if __name__ == "__main__":
    main()
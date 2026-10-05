from datetime import datetime
import logging
import os
from typing import Any, Dict, List, Optional
import requests
import ollama
import streamlit as st

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
LOG_FILE = "dashboard.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("GlobalIntelligenceApp")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Global Intelligence Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# API CLIENT LAYER
# ==========================================
class OpenAPIClient:
    """Handles communications with keyless public REST APIs with integrated logging."""

    @staticmethod
    def get_country_info(query_name: str) -> Optional[Dict[str, Any]]:
        query = query_name.strip()
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=en&format=json"
        
        logger.info(f"Initiating Geocoding lookup for query: '{query}'")
        try:
            start_time = datetime.now()
            response = requests.get(url, timeout=10)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Geocoding API responded with status {response.status_code} in {elapsed:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results")
                if results and len(results) > 0:
                    target = results[0]
                    resolved_info = {
                        "name": target.get("name", query_name),
                        "country": target.get("country", query_name),
                        "country_code": target.get("country_code", "N/A").upper(),
                        "population": target.get("population", 0),
                        "latlng": [
                            target.get("latitude", 0.0),
                            target.get("longitude", 0.0),
                        ],
                        "timezone": target.get("timezone", "UTC"),
                    }
                    logger.info(f"Successfully resolved '{query}' -> {resolved_info['name']}, {resolved_info['country']} ({resolved_info['latlng']})")
                    return resolved_info
                else:
                    logger.warning(f"No geocoding results found for query: '{query}'")
            else:
                logger.error(f"Geocoding API HTTP Error {response.status_code}: {response.text}")

        except requests.RequestException as e:
            logger.exception(f"Exception during Geocoding API request for '{query}': {e}")
            st.error(f"Geocoding API Connection Error: {e}")
            
        return None

    @staticmethod
    def get_weather_forecast(lat: float, lon: float) -> Optional[Dict[str, Any]]:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true"
        )
        logger.info(f"Fetching weather telemetry for coords: ({lat}, {lon})")
        try:
            start_time = datetime.now()
            response = requests.get(url, timeout=10)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Weather API responded with status {response.status_code} in {elapsed:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                current = data.get("current_weather", {})
                weather_info = {
                    "temperature": current.get("temperature", "N/A"),
                    "windspeed": current.get("windspeed", 0),
                    "weathercode": current.get("weathercode", 0),
                    "time": current.get("time", "N/A"),
                }
                logger.info(f"Weather metrics parsed: Temp={weather_info['temperature']}°C, Wind={weather_info['windspeed']} km/h")
                return weather_info
            else:
                logger.error(f"Weather API HTTP Error {response.status_code}: {response.text}")

        except requests.RequestException as e:
            logger.exception(f"Exception during Weather API request for ({lat}, {lon}): {e}")
            st.error(f"Weather API Error: {e}")
            
        return None

    @staticmethod
    def get_nearby_earthquakes(
        lat: float, lon: float, max_radius_km: float = 800
    ) -> List[Dict[str, Any]]:
        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&"
            f"latitude={lat}&longitude={lon}&maxradiuskm={max_radius_km}&limit=5"
        )
        logger.info(f"Querying USGS Earthquake API for radius {max_radius_km}km around ({lat}, {lon})")
        try:
            start_time = datetime.now()
            response = requests.get(url, timeout=10)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"USGS API responded with status {response.status_code} in {elapsed:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                features = data.get("features", [])
                events = []
                for feat in features:
                    props = feat.get("properties", {})
                    time_ms = props.get("time", 0)
                    formatted_time = (
                        datetime.fromtimestamp(time_ms / 1000).strftime(
                            "%Y-%m-%d %H:%M"
                        )
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
                logger.info(f"Parsed {len(events)} earthquake events near location")
                return events
            else:
                logger.error(f"USGS API HTTP Error {response.status_code}: {response.text}")

        except requests.RequestException as e:
            logger.exception(f"Exception during Earthquake API request: {e}")
            st.error(f"Earthquake API Error: {e}")
            
        return []


# ==========================================
# INTELLIGENCE & LLM LAYER
# ==========================================
class IntelligenceEngine:
    """Processes aggregated API data and generates AI analysis via Ollama."""

    @staticmethod
    def calculate_risk_score(
        weather: Dict[str, Any], earthquakes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        logger.info("Executing rule-based Risk Engine calculations")
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

        result = {
            "score": max(0, base_score),
            "level": risk_level,
            "factors": penalties if penalties else ["No major environmental risks"],
        }
        logger.info(f"Risk calculation complete: Score={result['score']}/100, Level={result['level']}")
        return result

    @staticmethod
    def generate_ai_briefing(
        location: Dict[str, Any],
        weather: Dict[str, Any],
        earthquakes: List[Dict[str, Any]],
        risk: Dict[str, Any],
        model_name: str,
    ) -> str:
        logger.info(f"Preparing LLM prompt for Ollama model '{model_name}'")
        prompt = f"""
        You are a senior geopolitical and environmental risk analyst. 
        Synthesize the following real-time telemetry into a concise, professional executive briefing (3-4 bullet points):

        Location Information:
        - Location: {location['name']} ({location['country']}, Code: {location['country_code']})
        - Population Estimate: {location['population']:,}
        - Timezone: {location['timezone']}

        Telemetry & Metrics:
        - Temperature: {weather['temperature']}°C
        - Wind Speed: {weather['windspeed']} km/h
        - Recent Nearby Earthquakes (800km radius): {len(earthquakes)} recorded events
        - Computed Safety Score: {risk['score']}/100 (Risk Level: {risk['level']})
        - Risk Factors: {', '.join(risk['factors'])}

        Focus on travel readiness, operational impacts, and safety recommendations.
        """

        try:
            start_time = datetime.now()
            logger.info(f"Invoking Ollama API ('{model_name}')...")
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Ollama generation successful in {elapsed:.2f}s")
            return response["message"]["content"]
        except Exception as e:
            logger.exception(f"Ollama invocation failed for model '{model_name}': {e}")
            return (
                f"⚠️ **Ollama AI Error:** Could not reach local model `{model_name}`.\n\n"
                f"**Details:** {e}\n\n"
                f"*Ensure `ollama serve` is running in your terminal and you have executed `ollama pull {model_name}`.*"
            )


# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
def main():
    logger.info("Initializing UI session")
    st.title("🌐 Global Intelligence & Risk Dashboard")
    st.caption(
        "Real-time telemetry aggregated from keyless public REST APIs & analyzed with local Ollama AI models."
    )

    # Sidebar Controls
    st.sidebar.header("🕹️ Control Panel")
    target_input = st.sidebar.text_input(
        "Target Location / Country", value="India", help="Enter a country or city name."
    )
    ollama_model = st.sidebar.text_input(
        "Ollama Model Name", value="llama3", help="Specify local Ollama model (e.g., llama3, mistral)."
    )
    search_button = st.sidebar.button("Generate Intelligence Report", type="primary")

    if search_button or target_input:
        logger.info(f"Dashboard run triggered for target: '{target_input}'")
        
        with st.spinner(f"Aggregating telemetry for '{target_input}'..."):
            location_data = OpenAPIClient.get_country_info(target_input)

            if not location_data:
                st.error(f"Could not resolve location coordinates for '{target_input}'. Please check spelling.")
                return

            lat, lon = location_data["latlng"][0], location_data["latlng"][1]

            # Fetch Telemetry
            weather_data = OpenAPIClient.get_weather_forecast(lat, lon)
            earthquake_data = OpenAPIClient.get_nearby_earthquakes(lat, lon)

            if not weather_data:
                st.error("Failed to fetch weather telemetry.")
                return

            # Compute Risk
            risk_metrics = IntelligenceEngine.calculate_risk_score(
                weather_data, earthquake_data
            )

        # ----------------------------------
        # METRICS ROW
        # ----------------------------------
        st.subheader(f"📍 Region: {location_data['name']}, {location_data['country']} ({location_data['country_code']})")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Population", f"{location_data['population']:,}")
        m2.metric("Temperature", f"{weather_data['temperature']} °C")
        m3.metric("Wind Speed", f"{weather_data['windspeed']} km/h")

        risk_color = (
            "off"
            if risk_metrics["level"] == "LOW"
            else "inverse"
            if risk_metrics["level"] == "HIGH"
            else "normal"
        )
        m4.metric(
            "Safety Score",
            f"{risk_metrics['score']}/100",
            delta=f"Risk: {risk_metrics['level']}",
            delta_color=risk_color,
        )

        st.divider()

        # ----------------------------------
        # DETAILED TELEMETRY LAYOUT
        # ----------------------------------
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### 📊 Environmental & Seismic Data")
            st.write(f"**Coordinates:** `{lat}°, {lon}°` | **Timezone:** `{location_data['timezone']}`")
            
            st.markdown("#### Key Risk Factors")
            for factor in risk_metrics["factors"]:
                st.warning(f"• {factor}") if risk_metrics["level"] != "LOW" else st.success(f"• {factor}")

            st.markdown("#### Recent Earthquakes (800 km Radius)")
            if earthquake_data:
                for eq in earthquake_data:
                    st.write(f"• **Mag {eq['mag']}** — {eq['place']} (*{eq['time']}*)")
            else:
                st.info("No recent seismic events recorded within 800 km.")

        with col_right:
            st.markdown("### 🤖 Ollama AI Executive Briefing")
            with st.spinner("Generating LLM intelligence synthesis..."):
                ai_briefing = IntelligenceEngine.generate_ai_briefing(
                    location_data,
                    weather_data,
                    earthquake_data,
                    risk_metrics,
                    model_name=ollama_model,
                )
            st.info(ai_briefing)

        # Map display
        st.divider()
        st.markdown("### 🗺️ Geographical Location")
        map_data = [{"lat": lat, "lon": lon}]
        st.map(map_data, zoom=4)

        # ----------------------------------
        # LIVE LOG VIEWER EXPANDER
        # ----------------------------------
        st.divider()
        with st.expander("🛠️ System Execution Logs (Live Debugging)", expanded=False):
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    log_lines = f.readlines()
                # Display the last 30 log entries
                recent_logs = "".join(log_lines[-30:])
                st.code(recent_logs, language="text")
            else:
                st.write("No log file generated yet.")


if __name__ == "__main__":
    main()
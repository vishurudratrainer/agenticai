
import requests

weather = requests.get(
    "https://api.open-meteo.com/v1/forecast?latitude=12.97&longitude=77.59&hourly=temperature_2m"
).json()

print("weatjer",)
prompt = f"Summarize this weather data: {weather['hourly']['temperature_2m'][:5]}"

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "mistral", "prompt": prompt},
    stream=True
)
import json
for line in resp.iter_lines():
    if line:
        print(json.loads(line.decode())["response"])

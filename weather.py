import json
import urllib.parse
import urllib.request


WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def get_weather(latitude, longitude):
    params = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "weather_code,"
            "wind_speed_10m"
        ),
        "timezone": "auto",
    })

    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.load(response)

    current = data["current"]

    code = int(current["weather_code"])

    return {
        "temperature_c": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "weather_code": code,
        "condition": WEATHER_CODES.get(code, "unknown"),
        "wind_kmh": current["wind_speed_10m"],
        "time": current["time"],
    }


def format_weather(latitude, longitude, location_name="current location"):
    try:
        weather = get_weather(latitude, longitude)

        return (
            f"Weather in {location_name}: "
            f"{weather['condition']}, "
            f"{weather['temperature_c']}°C, "
            f"humidity {weather['humidity']}%, "
            f"wind {weather['wind_kmh']} km/h."
        )

    except Exception as exc:
        return f"Weather service unavailable: {exc}"

import os
from huggingface_hub import InferenceClient
import requests

## You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type. If you run this on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN"
# powershell command: $env:HF_TOKEN = "xxxx"
# verify: echo $env:HF_TOKEN

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(model="moonshotai/Kimi-K2.5")

# output = client.chat.completions.create(
#     messages=[
#         {"role": "user", "content": "The capital of France is"},
#     ],
#     stream=False,
#     max_tokens=1024,
#     extra_body={'thinking': {'type': 'disabled'}},
# )
# print(output.choices[0].message.content)

# complex example

# This system prompt is a bit more complex and actually contains the function description already appended.
# Here we suppose that the textual description of the tools has already been appended.

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :

{{
  "action": "get_weather",
  "action_input": {{"location": "New York"}}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
]

#print(messages)

# output = client.chat.completions.create(
#     messages=messages,
#     stream=False,
#     max_tokens=200,
#     extra_body={'thinking': {'type': 'disabled'}},
# )
# print(output.choices[0].message.content)

# The answer was hallucinated by the model. We need to stop to actually execute the function!
output = client.chat.completions.create(
    messages=messages,
    max_tokens=150,
    stop=["Observation:"], # Let's stop before any actual function is called
    extra_body={'thinking': {'type': 'disabled'}},
)

print(output.choices[0].message.content)

# Dummy function
def get_weather(location):
    # 1) Geocode: location name -> lat/lon
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={location}&count=1&language=en&format=json"
    )
    geo_resp = requests.get(geo_url, timeout=20)
    geo_resp.raise_for_status()
    geo = geo_resp.json()

    # Ensure we have at least one result
    if not geo.get("results"):
        raise ValueError(f"No geocoding results for location: {location}")

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    # 2) Forecast: lat/lon -> weather
    forecast_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,wind_speed_10m"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    fc_resp = requests.get(forecast_url, timeout=20)
    fc_resp.raise_for_status()
    forecast = fc_resp.json()

    current = forecast.get("current", {})
    result = {
        "location": location,
        "latitude": lat,
        "longitude": lon,
        "temperature_2m": current.get("temperature_2m"),
        "wind_speed_10m": current.get("wind_speed_10m"),
        # optionally include hourly blocks:
        # "hourly": forecast.get("hourly", {})
    }

    return f"the weather in {location} is {current.get("temperature_2m")}degrees celsius with winds of {current.get("wind_speed_10m")}km/h. \n"

#get_weather('London')

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
    {"role": "assistant", "content": output.choices[0].message.content + "Observation:\n" + get_weather('London')},
]

output = client.chat.completions.create(
    messages=messages,
    stream=False,
    max_tokens=200,
    extra_body={'thinking': {'type': 'disabled'}},
)

print(output.choices[0].message.content)
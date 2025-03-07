import requests

from ragthoven.tools import BaseFunCalling


class WikipediaPageSearch(BaseFunCalling):
    def __init__(self) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.description = "Perform search of wikipedia for a given search term."
        self.parameters = {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "Entity for which a search is performed",
                }
            },
            "required": ["search_term"],
            "additionalProperties": False,
        }

    def __call__(self, args):
        return f"You have searched for: {args['search_term']}"


class WikipediaPageSummary(BaseFunCalling):
    def __init__(self) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.description = "Get summary of given wikipedia page"
        self.parameters = {
            "type": "object",
            "properties": {
                "page_title": {
                    "type": "string",
                    "description": "Entity for which a summary is retrieved",
                }
            },
            "required": ["page_title"],
            "additionalProperties": False,
        }

    def __call__(self, args):
        return f"You have requested summary of this wiki pade: {args['page_title']}"


class GetWeather(BaseFunCalling):
    def __init__(self) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.description = (
            "Get current temperature for provided coordinates in celsius."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False,
        }

    def get_weather(self, latitude, longitude):
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        data = response.json()
        return str(data["current"]["temperature_2m"])

    def __call__(self, args):
        return self.get_weather(args["latitude"], args["longitude"])

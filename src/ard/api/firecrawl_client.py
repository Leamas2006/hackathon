import requests

class FirecrawlClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "fc-d8b5246b3b744b1aa3d0a269948a5d08"

    def search(self, query: str, limit: int = 5):
        url = f"{self.base_url}/search"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "num_results": limit
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

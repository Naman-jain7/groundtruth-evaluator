import os
import requests
from pathlib import Path
import time
import json


OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://api.ollama.ai/v1")

MODELS = [
    os.getenv("MODEL_A", "mistral"),
    os.getenv("MODEL_B", "neural-chat"),
]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Batch settings
BATCH_SIZE = 10
TIMEOUT = 60  # seconds per request


class OllamaCloudClient:
    """Client for Ollama Cloud API with retry logic and error handling."""

    def __init__(self, api_key: str, base_url: str = OLLAMA_API_URL):
        if not api_key:
            raise ValueError(
                "OLLAMA_API_KEY not found in environment. ",
                "Please set it in your .env file.",
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        retries: int = 3,
    ) -> str:

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        for attempt in range(retries):
            try:
                response = self.session.post(
                    endpoint,
                    json=payload,
                    timeout=TIMEOUT,
                )
                response.raise_for_status()

                result = response.json()

                # Extract message content
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()

                raise ValueError(f"Unexpected API response format: {result}")

            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    print(
                        f"  Timeout on {model} (attempt {attempt + 1}/{retries}). ",
                        f"Retrying in {wait_time}s...",
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"API timeout after {retries} retries for model {model}"
                    )

            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"  Request error on {model} (attempt {attempt + 1}/{retries}): ",
                        f"{str(e)[:100]}. Retrying in {wait_time}s...",
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"API request failed after {retries} retries: {str(e)}"
                    )

            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse API response: {str(e)}")

        raise RuntimeError(f"Exhausted all retries for model {model}")

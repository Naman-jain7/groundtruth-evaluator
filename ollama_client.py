import time

from ollama import Client

from config import OLLAMA_API_KEY, OLLAMA_API_URL


class OllamaCloudClient:
    """Client for Ollama Cloud API with retry logic and error handling."""

    def __init__(self):  # type:ignore
        self.api_key = OLLAMA_API_KEY
        self.api_url = OLLAMA_API_URL
        self.client = Client(
            host=self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def generate(
        self,
        model: str,
        prompt: str,
        system_message: str = "Please provide a concise response (in 1-3 lines).",
        max_tokens: int = 512,
        temperature: float = 0.7,
        retries: int = 5,
    ) -> str:
        """Generate a response using the Ollama Cloud API (same approach as trial())."""

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(retries):
            try:
                # Use the ollama library client (same as trial()) with streaming
                chunks = []
                for chunk in self.client.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    options=options,
                ):
                    chunks.append(chunk['message']['content'])

                return "".join(chunks).strip()

            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 3 ** attempt  # Exponential backoff: 1s, 3s, 9s, 27s
                    print(
                        f"  Error on {model} (attempt {attempt + 1}/{retries}): "
                        f"{str(e)[:100]}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"API request failed after {retries} retries: {str(e)}"
                    )

        raise RuntimeError(f"Exhausted all retries for model {model}")

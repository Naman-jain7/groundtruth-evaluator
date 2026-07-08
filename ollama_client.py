import asyncio
import json
from typing import Any

import requests
from ollama import Client
from config import AVAILABLE_MODELS, OPENROUTER_API_KEY_1, OPENROUTER_API_URL


class OllamaLocalClient:
    """Client for local Ollama generation."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")
        self.generate_url = f"{self.host}/api/generate"
        self.available_models = {model for model in AVAILABLE_MODELS if model}

    def generate(
        self,
        model: str,
        prompt: str,
        system_message: str = (
            "You are a precise and truthful assistant. Provide direct, objective, and accurate "
            "answers in 1-2 sentences, avoiding speculation or common misconceptions."
        ),
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate an answer from a local Ollama model."""

        if not model:
            raise ValueError("Model name is required for local generation.")
        if self.available_models and model not in self.available_models:
            raise ValueError(f"Model {model} is not listed in AVAILABLE_MODELS.")

        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_message,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                stream=False,
                timeout=120,
            )
            response.raise_for_status()

            chunks = []
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                chunks.append(data.get("response", ""))

            return "".join(chunks).strip()
        
        except requests.RequestException as e:
            raise RuntimeError(f"Local Ollama request failed for model {model}: {e}") from e
        
        except (json.JSONDecodeError, RuntimeError) as e:
            raise RuntimeError(f"Local Ollama generation failed for model {model}: {e}") from e


class OpenRouterCloudClient:
    """Client for OpenRouter Cloud API using langchain-openai."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or OPENROUTER_API_KEY_1
        self.api_url = OPENROUTER_API_URL
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for cloud evaluation.")

        self.available_models = {model for model in AVAILABLE_MODELS if model}

    def _validate_schema_response(self, schema: type[Any], response: Any) -> Any:
        # ChatOpenAI's with_structured_output returns the parsed object if it matches the schema (Pydantic model)
        if isinstance(response, schema):
            return response
        
        # Fallback if somehow it returns a dict or string
        if isinstance(response, dict):
            return schema(**response)
        
        if isinstance(response, str):
            try:
                if hasattr(schema, "model_validate_json"):
                    return schema.model_validate_json(response)
                return schema.parse_raw(response)
            except Exception:
                return response
                
        return response

    def generate(
        self,
        model: str,
        prompt: str,
        system_message: str = "Judge strictly. Return only valid JSON. No markdown.",
        max_tokens: int = 512,
        temperature: float = 0.0,
        schema: type[Any] | None = None,
    ) -> Any:
        """Generate an evaluation response using the OpenRouter API via ChatOpenAI."""

        if not model:
            raise ValueError("Model name is required for cloud evaluation.")
        
        if self.available_models and model not in self.available_models:
            raise ValueError(f"Model {model} is not listed in AVAILABLE_MODELS.")

        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            base_url=self.api_url,
            api_key=self.api_key, # type: ignore
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens
        )

        messages = []
        if system_message:
            messages.append(("system", system_message))
        messages.append(("user", prompt))

        try:
            if schema is not None:
                llm_with_schema = llm.with_structured_output(schema)
                response = llm_with_schema.invoke(messages)
                return self._validate_schema_response(schema, response)
            else:
                response = llm.invoke(messages)
                return response.content
                
        except Exception as e:
            raise RuntimeError(f"API request failed for model {model}: {e}") from e

    async def a_generate(
        self,
        model: str,
        prompt: str,
        system_message: str = "Return only valid JSON. No markdown.",
        max_tokens: int = 512,
        temperature: float = 0.0,
        schema: type[Any] | None = None,
    ) -> Any:
        """Async wrapper for cloud evaluation generation."""

        return await asyncio.to_thread(
            self.generate,
            model=model,
            prompt=prompt,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=schema,
        )


class OllamaCloudClient:
    """Client for Ollama Cloud API with error handling."""

    def __init__(self, api_key: str | None = None):  # type:ignore
        self.api_key = api_key or OPENROUTER_API_KEY_1
        self.api_url = OPENROUTER_API_URL
        if not self.api_key:
            raise ValueError("OLLAMA_API_KEY is required for cloud evaluation.")
        if not self.api_url:
            raise ValueError("OLLAMA_API_URL is required for cloud evaluation.")

        self.client = Client(
            host=self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        self.available_models = {model for model in AVAILABLE_MODELS if model}

    def _schema_json(self, schema: type[Any]) -> dict[str, Any]:
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        return schema.schema()

    def _validate_schema_response(self, schema: type[Any], response: str) -> Any:
        try:
            if hasattr(schema, "model_validate_json"):
                return schema.model_validate_json(response)
            return schema.parse_raw(response)
        except Exception:
            return response

    def generate(
        self,
        model: str,
        prompt: str,
        system_message: str = "Judge strictly. Return only valid JSON. No markdown.",
        max_tokens: int = 512,
        temperature: float = 0.7,
        schema: type[Any] | None = None,
    ) -> Any:
        """Generate an evaluation response using the Ollama Cloud API."""

        if not model:
            raise ValueError("Model name is required for cloud evaluation.")
        
        if self.available_models and model not in self.available_models:
            raise ValueError(f"Model {model} is not listed in AVAILABLE_MODELS.")

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
        }
        if schema is not None:
            chat_kwargs["format"] = self._schema_json(schema)

        try:
            # Use the ollama library client (same as trial()) with streaming.
            chunks = []
            for chunk in self.client.chat(**chat_kwargs):
                chunks.append(chunk["message"]["content"])

            response = "".join(chunks).strip()
            if schema is not None:
                return self._validate_schema_response(schema, response)
            return response
        except Exception as e:
            raise RuntimeError(f"API request failed for model {model}: {e}") from e

    async def a_generate(
        self,
        model: str,
        prompt: str,
        system_message: str = "Return only valid JSON. No markdown.",
        max_tokens: int = 512,
        temperature: float = 0.7,
        schema: type[Any] | None = None,
    ) -> Any:
        """Async wrapper for cloud evaluation generation."""

        return await asyncio.to_thread(
            self.generate,
            model=model,
            prompt=prompt,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            schema=schema,
        )
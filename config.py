import os
from pathlib import Path

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

MODELS = [
    os.getenv("MODEL_A", "mistral"),
    os.getenv("MODEL_B", "neural-chat"),
]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

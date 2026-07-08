import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")

OPENROUTER_API_KEY_1 = os.getenv("OPENROUTER_API_KEY_1")
OPENROUTER_API_KEY_2 = os.getenv("OPENROUTER_API_KEY_2")
OPENROUTER_API_KEY_3 = os.getenv("OPENROUTER_API_KEY_3")

AVAILABLE_MODELS = [
    os.getenv("MODEL_A"),
    os.getenv("MODEL_B"),
    os.getenv("EVAL_MODEL"),
]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_ANSWERS_PATH = OUTPUT_DIR / "raw_answers.csv"

EVALUATED_RESULTS_PATH = OUTPUT_DIR / "evaluated_results.csv"

from typing import Optional
from datasets import load_from_disk

def load_truthful_qa(split: str = "generation", limit: Optional[int] = None):
    """
    Load TruthfulQA dataset from Hugging Face.

    Args:
        split: Dataset split ('generation' or 'validation')
        limit: Maximum number of samples to load

    Returns:
        List of dictionaries with 'question', 'category', 'correct_answers'
    """
    print(f"Loading TruthfulQA from disk...")
    try:
        dataset = load_from_disk("data/truthful_qa")

        samples = []
        for idx, item in enumerate(dataset):
            if limit and idx >= limit:
                break

            samples.append(
                {
                    "question_id": f"{split}_{idx:04d}",
                    "question": item["question"],  # type: ignore
                    "category": item.get("category", "unknown"),  # type: ignore
                    "correct_answers": item.get("correct_answers", []),  # type: ignore
                }
            )

        print(f"✅ Loaded {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

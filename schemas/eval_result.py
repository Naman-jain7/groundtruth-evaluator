from dataclasses import dataclass
from typing import Optional

@dataclass
class EvaluationResult:
    """Store evaluation results for a single question-answer pair."""

    question_id: str
    question: str
    category: str
    model: str
    answer: str
    correct_answer: str
    hallucination_score: Optional[float] = None
    hallucination_metric: Optional[bool] = None
    answer_relevancy_score: Optional[float] = None
    answer_relevancy_metric: Optional[bool] = None
    faithfulness_score: Optional[float] = None
    faithfulness_metric: Optional[bool] = None
    error: Optional[str] = None
    timestamp: str = ""

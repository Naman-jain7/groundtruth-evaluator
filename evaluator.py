import asyncio
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from config import OPENROUTER_API_KEY_1, OPENROUTER_API_KEY_2, OPENROUTER_API_KEY_3
from ollama_client import OpenRouterCloudClient

load_dotenv()

class DeepEvalOpenRouterModel(DeepEvalBaseLLM):
    """Custom LLM wrapper for DeepEval using OpenRouter Cloud Client."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenRouterCloudClient(api_key=api_key)

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: type[Any] | None = None) -> Any:
        return self.client.generate(
            model=self.model_name,
            prompt=prompt,
            system_message="Return only valid JSON. No markdown.",
            schema=schema,
        )

    async def a_generate(
        self,
        prompt: str,
        schema: type[Any] | None = None,
    ) -> Any:
        return await self.client.a_generate(
            model=self.model_name,
            prompt=prompt,
            system_message="Return only valid JSON. No markdown.",
            schema=schema,
        )

class DeepEvalEvaluator:
    """Wrapper for DeepEval metrics with error handling."""

    def __init__(self, eval_model_name):
        if not OPENROUTER_API_KEY_1 or not OPENROUTER_API_KEY_2 or not OPENROUTER_API_KEY_3:
            raise ValueError(
                "OLLAMA_API_KEY_1, OLLAMA_API_KEY_2, and OLLAMA_API_KEY_3 "
                "are required for async metric evaluation."
            )

        self.model_1 = DeepEvalOpenRouterModel(
            model_name=eval_model_name,
            api_key=OPENROUTER_API_KEY_1,
        )
        self.model_2 = DeepEvalOpenRouterModel(
            model_name=eval_model_name,
            api_key=OPENROUTER_API_KEY_2,
        )
        self.model_3 = DeepEvalOpenRouterModel(
            model_name=eval_model_name,
            api_key=OPENROUTER_API_KEY_3,
        )

        self.hallucination_metric = HallucinationMetric(
            threshold=0.5,
            model=self.model_1,
        )
        self.relevancy_metric = AnswerRelevancyMetric(
            threshold=0.5,
            model=self.model_2,
        )
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=0.5,
            model=self.model_3,
        )

    def _measure_metric(
        self,
        metric,
        test_case: LLMTestCase,
        score_key: str,
        pass_key: str,
        label: str,
    ) -> dict:
        try:
            metric.measure(test_case)
            return {
                score_key: metric.score,
                pass_key: metric.is_successful(),
            }
        except Exception as e:
            print(f"    ⚠️  {label} metric error: {str(e)[:80]}")
            return {
                score_key: None,
                pass_key: None,
            }

    async def evaluate_async(self, question: str, answer: str, context: str) -> dict:
        """Evaluate all metrics sequentially using OpenRouter."""

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=[context],
            retrieval_context=[context],
        )

        results = {}
        
        # Run sequentially to avoid rate limits
        results.update(
            await asyncio.to_thread(
                self._measure_metric,
                self.hallucination_metric,
                test_case,
                "hallucination_score",
                "hallucination_pass",
                "Hallucination",
            )
        )
        
        results.update(
            await asyncio.to_thread(
                self._measure_metric,
                self.relevancy_metric,
                test_case,
                "relevancy_score",
                "relevancy_pass",
                "Relevancy",
            )
        )
        
        results.update(
            await asyncio.to_thread(
                self._measure_metric,
                self.faithfulness_metric,
                test_case,
                "faithfulness_score",
                "faithfulness_pass",
                "Faithfulness",
            )
        )

        return results

    def evaluate(self, question: str, answer: str, context: str) -> dict:
        """
        Evaluate answer using all three metrics.

        Args:
            question: Input question
            answer: Model-generated answer
            context: Ground truth answer (serves as context)

        Returns:
            Dictionary with metric scores and pass/fail status
        """
        try:
            return asyncio.run(self.evaluate_async(question, answer, context))

        except Exception as e:
            print(f"    ❌ Evaluation failed: {str(e)[:80]}")
            return {
                "hallucination_score": None,
                "hallucination_pass": None,
                "relevancy_score": None,
                "relevancy_pass": None,
                "faithfulness_score": None,
                "faithfulness_pass": None,
            }
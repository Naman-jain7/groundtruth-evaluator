import os

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from ollama_client import OllamaCloudClient

load_dotenv()

class DeepEvalOllamaModel(DeepEvalBaseLLM):
    """Custom LLM wrapper for DeepEval using Ollama Cloud Client."""
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OllamaCloudClient(api_key)

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str) -> str:
        # Prepend a system message to instruct the judge model to be concise and accurate
        return self.client.generate(
            model=self.model_name,
            prompt=prompt,
            system_message="You are an evaluation judge. Return your evaluation strictly in the requested format."
        )

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

class DeepEvalEvaluator:
    """Wrapper for DeepEval metrics with error handling."""

    def __init__(self, eval_model_name):
        api_key = os.getenv("OLLAMA_API_KEY")
        eval_model_name = eval_model_name
        
        if api_key:
            self.model = DeepEvalOllamaModel(
                model_name=eval_model_name,
                api_key=api_key,
            )
        else:
            self.model = None

        self.hallucination_metric = HallucinationMetric(threshold=0.5, model=self.model)
        self.relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=self.model)
        self.faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=self.model)

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
            # Create test case for evaluation
            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                context=[context],  # Ground truth as context
                retrieval_context=[context],  # Required by FaithfulnessMetric
            )

            # Evaluate with each metric
            results = {}

            # Hallucination metric
            try:
                self.hallucination_metric.measure(test_case)
                results["hallucination_score"] = self.hallucination_metric.score
                results["hallucination_pass"] = (
                    self.hallucination_metric.is_successful()
                )
            except Exception as e:
                print(f"    ⚠️  Hallucination metric error: {str(e)[:80]}")
                results["hallucination_score"] = None
                results["hallucination_pass"] = None

            # Answer relevancy metric
            try:
                self.relevancy_metric.measure(test_case)
                results["relevancy_score"] = self.relevancy_metric.score
                results["relevancy_pass"] = self.relevancy_metric.is_successful()
            except Exception as e:
                print(f"    ⚠️  Relevancy metric error: {str(e)[:80]}")
                results["relevancy_score"] = None
                results["relevancy_pass"] = None

            # Faithfulness metric
            try:
                self.faithfulness_metric.measure(test_case)
                results["faithfulness_score"] = self.faithfulness_metric.score
                results["faithfulness_pass"] = self.faithfulness_metric.is_successful()
            except Exception as e:
                print(f"    ⚠️  Faithfulness metric error: {str(e)[:80]}")
                results["faithfulness_score"] = None
                results["faithfulness_pass"] = None

            return results

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

"""
Main Pipeline Orchestrator
Coordinates: answers → evaluate → aggregate → plot
"""

import argparse
import os
import sys
from typing import Optional

# Import pipeline modules
from generate_answers import generate_answers
from evaluate_answers import evaluate_answers
from aggregate_categories import aggregate_categories
from generate_plots import generate_plots_and_reports
from config import OUTPUT_DIR, AVAILABLE_MODELS


def run_full_pipeline(
    eval_model_name: str,
    models: list,
    dataset_limit: Optional[int] = None,
    sample_questions: int = 0,
    skip_generation: bool = True
):
    """
    Run the complete 4-stage evaluation pipeline:
      Stage 1: Generate raw answers (local Ollama)
      Stage 2: Evaluate answers (OpenRouter / DeepEval)
      Stage 3: Aggregate by category
      Stage 4: Generate plots & reports
    """
    print("\n" + "🚀 " * 40)
    print("\n" + "=" * 80)
    print("TRUTHFUL_QA EVALUATION PIPELINE")
    print("=" * 80 + "\n")

    # Stage 1: Generate raw answers
    print("\n[STAGE 1/4] GENERATING MODEL ANSWERS")
    print("-" * 80)
    try:
        results, raw_path = generate_answers(
            models=models,
            dataset_limit=dataset_limit,
            sample_questions=sample_questions,
        )
        if raw_path is None:
            print("❌ Stage 1 failed: no output produced")
            return False
        print(f"✅ Stage 1 complete: {raw_path}")
    except Exception as e:
        print(f"❌ Stage 1 failed: {e}")
        return False

    # Stage 2: Evaluate answers
    print("\n[STAGE 2/4] EVALUATING ANSWERS")
    print("-" * 80)
    try:
        eval_results, eval_path = evaluate_answers(
            eval_model_name=eval_model_name,
        )
        if eval_path is None:
            print("❌ Stage 2 failed: no output produced")
            return False
        print(f"✅ Stage 2 complete: {eval_path}")
    except Exception as e:
        print(f"❌ Stage 2 failed: {e}")
        return False

    # Stage 3: Aggregate by category
    print("\n[STAGE 3/4] AGGREGATING BY CATEGORY")
    print("-" * 80)
    try:
        agg_results = aggregate_categories(eval_path)  # noqa: F841
        print("✅ Stage 3 complete")
    except Exception as e:
        print(f"❌ Stage 3 failed: {e}")
        return False

    # Stage 4: Generate plots and reports
    print("\n[STAGE 4/4] GENERATING PLOTS & REPORTS")
    print("-" * 80)
    try:
        plots, report = generate_plots_and_reports()  # type: ignore
        print("✅ Stage 4 complete")
    except Exception as e:
        print(f"❌ Stage 4 failed: {e}")
        return False

    # Final summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    print("\n📁 OUTPUT FILES:")
    print(f"  Raw Answers:         {OUTPUT_DIR / 'raw_answers.csv'}")
    print(f"  Evaluated Results:   {eval_path}")
    print(f"  Category Aggregated: {OUTPUT_DIR / 'category_aggregated.csv'}")
    print(f"  Summary Statistics:  {OUTPUT_DIR / 'category_summary_stats.json'}")
    print(f"  Comparison Report:   {report}")
    print(f"  Plots Directory:     {OUTPUT_DIR / 'plots'}/")
    print("\n" + "🎉 " * 40 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="TruthfulQA Evaluation Pipeline with DeepEval"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Number of questions to use (0 = all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit dataset size (for testing)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["answers", "evaluate", "aggregate", "plot", "all"],
        default="all",
        help=(
            "Stage to run:\n"
            "  answers   - Phase 1: generate raw answers with local Ollama models\n"
            "  evaluate  - Phase 2: evaluate raw_answers.csv with DeepEval\n"
            "  aggregate - Aggregate evaluated_results.csv by category\n"
            "  plot      - Generate plots & reports\n"
            "  all       - Run all 4 stages in sequence (default)"
        ),
    )

    args = parser.parse_args()

    eval_model_cli = os.getenv("EVAL_MODEL", "openai/gpt-oss-120b:free")
    success = False

    if args.stage == "all":
        success = run_full_pipeline(
            eval_model_name=eval_model_cli,
            models=AVAILABLE_MODELS,
            dataset_limit=args.limit,
            sample_questions=args.sample,
        )

    elif args.stage == "answers":
        print("\n[STAGE 1] GENERATING MODEL ANSWERS")
        print("-" * 80)
        results, raw_path = generate_answers(
            models=AVAILABLE_MODELS,
            dataset_limit=args.limit,
            sample_questions=args.sample,
        )
        success = raw_path is not None and raw_path.exists()
        if success:
            print(f"✅ Raw answers saved to: {raw_path}")

    elif args.stage == "evaluate":
        print("\n[STAGE 2] EVALUATING ANSWERS")
        print("-" * 80)
        eval_results, eval_path = evaluate_answers(
            eval_model_name=eval_model_cli,
        )
        success = eval_path is not None and eval_path.exists()
        if success:
            print(f"✅ Evaluated results saved to: {eval_path}")

    elif args.stage == "aggregate":
        print("\n[STAGE 3] AGGREGATING BY CATEGORY")
        print("-" * 80)
        evaluated_path = OUTPUT_DIR / "evaluated_results.csv"
        if not evaluated_path.exists():
            print(f"❌ Evaluated results not found: {evaluated_path}")
            print("   Run evaluate stage first: python main.py --stage evaluate")
            return 1
        aggregate_categories(evaluated_path)
        success = True

    elif args.stage == "plot":
        print("\n[STAGE 4] GENERATING PLOTS & REPORTS")
        print("-" * 80)
        generate_plots_and_reports()
        success = True

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

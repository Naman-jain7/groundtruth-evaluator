"""
Main Pipeline Orchestrator
Coordinates: output generation → aggregation → plotting → reporting
"""

import argparse
import os
import sys
from typing import Optional

# Import pipeline modules
from generate_outputs import generate_outputs
from aggregate_categories import aggregate_categories
from generate_plots import generate_plots_and_reports
from config import OUTPUT_DIR, MODELS

def run_full_pipeline(
    eval_model_name: str,
    models: list,
    dataset_limit: Optional[int] = None,
    sample_questions: int = 0,
    skip_generation: bool = False,
):
    """
    Run the complete evaluation pipeline.

    Args:
        dataset_limit: Limit dataset size (for testing)
        sample_questions: If > 0, only evaluate this many questions
        skip_generation: Skip output generation, use existing CSV
    """
    print("\n" + "🚀 " * 40)
    print("\n" + "=" * 80)
    print("TRUTHFUL_QA EVALUATION PIPELINE")
    print("=" * 80 + "\n")

    # Stage 1: Generate outputs
    if not skip_generation:
        print("\n[STAGE 1/3] GENERATING MODEL OUTPUTS")
        print("-" * 80)
        try:
            results, csv_path = generate_outputs(
                eval_model_name=eval_model_name,
                models=models,
                dataset_limit=dataset_limit,
                sample_questions=sample_questions,
            ) # type: ignore
            print(f"✅ Stage 1 complete: {csv_path}")
        except Exception as e:
            print(f"❌ Stage 1 failed: {e}")
            return False
    else:
        csv_path = OUTPUT_DIR / "evaluation_results.csv"
        if not csv_path.exists():
            print(f"❌ CSV file not found: {csv_path}")
            return False
        print(f"⏭️  Skipping generation, using existing: {csv_path}")

    # Stage 2: Aggregate by category
    print("\n[STAGE 2/3] AGGREGATING BY CATEGORY")
    print("-" * 80)
    try:
        agg_results = aggregate_categories(csv_path)  # noqa: F841
        print("✅ Stage 2 complete")
    except Exception as e:
        print(f"❌ Stage 2 failed: {e}")
        return False

    # Stage 3: Generate plots and reports
    print("\n[STAGE 3/3] GENERATING PLOTS & REPORTS")
    print("-" * 80)
    try:
        plots, report = generate_plots_and_reports() # type: ignore
        print("✅ Stage 3 complete")
    except Exception as e:
        print(f"❌ Stage 3 failed: {e}")
        return False

    # Final summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    print("\n📁 OUTPUT FILES:")
    print(f"  Evaluation Results:  {csv_path}")
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
        help="Number of questions to evaluate (0 = all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit dataset size (for testing)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip output generation, use existing CSV",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["generate", "aggregate", "plot", "all"],
        default="all",
        help="Run specific stage or all stages",
    )

    args = parser.parse_args()

    # Use env vars for CLI defaults if not provided by app
    eval_model_cli = os.getenv("EVAL_MODEL", "mistral")
    
    # Route to appropriate pipeline
    if args.stage == "all":
        success = run_full_pipeline(
            eval_model_name=eval_model_cli,
            models=MODELS,
            dataset_limit=args.limit,
            sample_questions=args.sample,
            skip_generation=args.skip_generation,
        )
    elif args.stage == "generate":
        print("\n[STAGE 1/3] GENERATING MODEL OUTPUTS")
        print("-" * 80)
        results, csv_path = generate_outputs(
            eval_model_name=eval_model_cli,
            models=MODELS,
            dataset_limit=args.limit,
            sample_questions=args.sample,
        ) # type: ignore
        success = csv_path.exists()
    elif args.stage == "aggregate":
        print("\n[STAGE 2/3] AGGREGATING BY CATEGORY")
        print("-" * 80)
        aggregate_categories()
        success = True
    elif args.stage == "plot":
        print("\n[STAGE 3/3] GENERATING PLOTS & REPORTS")
        print("-" * 80)
        generate_plots_and_reports()
        success = True

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

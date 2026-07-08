"""
Category Aggregation: Group evaluation results by topic category.
Calculates hallucination rate, answer relevancy, and faithfulness metrics per category.
"""

import json
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("outputs")

METRICS = [
    "hallucination_metric",
    "answer_relevancy_metric",
    "faithfulness_metric",
]

SCORE_METRICS = [
    "hallucination_score",
    "answer_relevancy_score",
    "faithfulness_score",
]


# ============================================================================
# Aggregation Functions
# ============================================================================


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load evaluation results from CSV."""
    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} records\n")
    return df


def aggregate_by_category(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group results by category and calculate aggregate metrics.

    Returns:
        Dictionary mapping category -> aggregated results DataFrame
    """
    categories = df["category"].unique()
    aggregated = {}

    print(f"Aggregating by category (n={len(categories)})...\n")

    for category in sorted(categories):
        cat_data = df[df["category"] == category]

        # Group by model
        models = cat_data["model"].unique()
        category_results = []

        for model in sorted(models):
            model_data = cat_data[cat_data["model"] == model]
            model_data_clean = model_data.dropna(
                subset=[
                    "hallucination_metric",
                    "answer_relevancy_metric",
                    "faithfulness_metric",
                ]
            )

            if len(model_data_clean) == 0:
                continue

            # Calculate pass rates (percentage of True values)
            hallucination_rate = (
                (model_data_clean["hallucination_metric"].astype(bool).sum())
                / len(model_data_clean)
                * 100
            )
            relevancy_rate = (
                (model_data_clean["answer_relevancy_metric"].astype(bool).sum())
                / len(model_data_clean)
                * 100
            )
            faithfulness_rate = (
                (model_data_clean["faithfulness_metric"].astype(bool).sum())
                / len(model_data_clean)
                * 100
            )

            # Calculate average scores
            hallucination_avg = model_data_clean["hallucination_score"].mean()
            relevancy_avg = model_data_clean["answer_relevancy_score"].mean()
            faithfulness_avg = model_data_clean["faithfulness_score"].mean()

            category_results.append(
                {
                    "category": category,
                    "model": model,
                    "sample_count": len(model_data_clean),
                    "hallucination_pass_rate": hallucination_rate,
                    "hallucination_avg_score": hallucination_avg,
                    "answer_relevancy_pass_rate": relevancy_rate,
                    "answer_relevancy_avg_score": relevancy_avg,
                    "faithfulness_pass_rate": faithfulness_rate,
                    "faithfulness_avg_score": faithfulness_avg,
                }
            )

        if category_results:
            aggregated[category] = pd.DataFrame(category_results)
            print(
                f"  {category:20s} → {len(category_results)} model(s), "
                f"{sum(r['sample_count'] for r in category_results)} samples"
            )

    print(f"\n✅ Aggregated {len(aggregated)} categories\n")
    return aggregated


def calculate_hallucination_rate(df: pd.DataFrame, by_category: bool = True) -> dict:
    """
    Calculate hallucination rate: % of responses that PASS hallucination check
    (lower hallucination = higher pass rate = better).

    Args:
        df: Full evaluation DataFrame
        by_category: If True, group by category; else overall

    Returns:
        Dictionary with hallucination statistics
    """
    if by_category:
        result = {}
        for category in sorted(df["category"].unique()):
            cat_data = df[df["category"] == category]
            for model in sorted(cat_data["model"].unique()):
                model_data = cat_data[cat_data["model"] == model].dropna(
                    subset=["hallucination_metric"]
                )
                if len(model_data) > 0:
                    rate = (
                        (model_data["hallucination_metric"].astype(bool).sum())
                        / len(model_data)
                        * 100
                    )
                    key = f"{category}_{model}"
                    result[key] = {
                        "category": category,
                        "model": model,
                        "hallucination_pass_rate": rate,
                        "sample_count": len(model_data),
                    }
        return result
    else:
        clean_df = df.dropna(subset=["hallucination_metric"])
        rate = (
            (clean_df["hallucination_metric"].astype(bool).sum()) / len(clean_df) * 100
        )
        return {"overall_hallucination_pass_rate": rate}


# ============================================================================
# Export Functions
# ============================================================================


def export_aggregated_results(
    aggregated: Dict[str, pd.DataFrame], output_dir: Path = OUTPUT_DIR
):
    """Export aggregated results to JSON and CSV files."""
    output_dir.mkdir(exist_ok=True)

    # Combined view (all categories stacked)
    combined_df = pd.concat(aggregated.values(), ignore_index=True)
    combined_path = output_dir / "category_aggregated.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"✅ Saved combined aggregation: {combined_path}")

    # Per-category JSON
    aggregated_json = {}
    for category, cat_df in aggregated.items():
        aggregated_json[category] = cat_df.to_dict(orient="records")

    json_path = output_dir / "category_aggregated.json"
    with open(json_path, "w") as f:
        json.dump(aggregated_json, f, indent=2, cls=NumpyEncoder)
    print(f"✅ Saved JSON aggregation: {json_path}")

    # Summary statistics
    summary_stats = generate_summary_stats(combined_df)
    summary_path = output_dir / "category_summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2, cls=NumpyEncoder)
    print(f"✅ Saved summary statistics: {summary_path}\n")

    return combined_path, json_path, summary_path


def generate_summary_stats(df: pd.DataFrame) -> dict:
    """Generate high-level summary statistics."""
    summary = {}

    # By category
    for category in df["category"].unique():
        cat_data = df[df["category"] == category]
        summary[category] = {
            "models_evaluated": len(cat_data["model"].unique()),
            "avg_hallucination_pass_rate": cat_data["hallucination_pass_rate"].mean(),
            "avg_relevancy_pass_rate": cat_data["answer_relevancy_pass_rate"].mean(),
            "avg_faithfulness_pass_rate": cat_data["faithfulness_pass_rate"].mean(),
            "total_samples": cat_data["sample_count"].sum(),
        }

    # Overall
    summary["overall"] = {
        "total_categories": len(df["category"].unique()),
        "total_models": len(df["model"].unique()),
        "total_evaluations": df["sample_count"].sum(),
        "avg_hallucination_pass_rate": df["hallucination_pass_rate"].mean(),
        "avg_relevancy_pass_rate": df["answer_relevancy_pass_rate"].mean(),
        "avg_faithfulness_pass_rate": df["faithfulness_pass_rate"].mean(),
    }

    return summary



def aggregate_categories(csv_path: Path = None): # type: ignore
    """Main aggregation pipeline."""
    print("\n" + "=" * 80)
    print("Category Aggregation Pipeline")
    print("=" * 80 + "\n")

    if csv_path is None:
        csv_path = OUTPUT_DIR / "evaluated_results.csv"

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return

    # Load and aggregate
    df = load_results(csv_path)
    aggregated = aggregate_by_category(df)

    # Calculate hallucination rates
    hallucination_by_category = calculate_hallucination_rate(df, by_category=True)
    print("Hallucination Pass Rates by Category & Model:")
    for key, stats in sorted(hallucination_by_category.items()):
        print(
            f"  {key:40s}: {stats['hallucination_pass_rate']:6.2f}% "
            f"(n={stats['sample_count']})"
        )
    print()

    # Export results
    combined_path, json_path, summary_path = export_aggregated_results(aggregated)

    print("=" * 80)
    print("Category Aggregation Complete!")
    print("=" * 80 + "\n")

    return {
        "aggregated": aggregated,
        "hallucination_rates": hallucination_by_category,
        "combined_csv": combined_path,
        "json_file": json_path,
        "summary_stats": summary_path,
    }


if __name__ == "__main__":
    aggregate_categories()

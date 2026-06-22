"""
Generate plots and model comparison reports from aggregated evaluation results.
Visualizations include:
  - Hallucination rate by category
  - Model comparison (cross-model metrics)
  - Metric distributions
  - Category performance heatmaps
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10
COLORS = sns.color_palette("husl", 8)


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_hallucination_rate_by_category(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Plot hallucination pass rate grouped by category and model.
    Higher rate = fewer hallucinations = better performance.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare data
    plot_data = df.copy()
    plot_data = plot_data.sort_values(["category", "hallucination_pass_rate"], ascending=[True, False])

    # Create grouped bar chart
    categories = plot_data["category"].unique()
    models = plot_data["model"].unique()
    x = np.arange(len(categories))
    width = 0.35

    for i, model in enumerate(models):
        model_data = plot_data[plot_data["model"] == model]
        # Align data to categories
        values = []
        for cat in categories:
            cat_model = model_data[model_data["category"] == cat]
            if len(cat_model) > 0:
                values.append(cat_model["hallucination_pass_rate"].values[0])
            else:
                values.append(0)

        offset = (i - len(models) / 2) * width
        ax.bar(x + offset, values, width, label=model, color=COLORS[i])

    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Hallucination Pass Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Hallucination Pass Rate by Category & Model\n(Higher is Better)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend(title="Model", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path is None:
        output_path = PLOTS_DIR / "hallucination_by_category.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")
    return output_path


def plot_model_comparison(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Create a comprehensive model comparison across all metrics.
    """
    models = df["model"].unique()
    metrics = [
        ("hallucination_pass_rate", "Hallucination Pass Rate"),
        ("answer_relevancy_pass_rate", "Answer Relevancy Pass Rate"),
        ("faithfulness_pass_rate", "Faithfulness Pass Rate"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (metric_col, metric_name) in enumerate(metrics):
        ax = axes[idx]

        # Calculate mean score per model
        model_scores = []
        for model in sorted(models):
            model_df = df[df["model"] == model]
            score = model_df[metric_col].mean()
            model_scores.append({"model": model, "score": score})

        model_df_plot = pd.DataFrame(model_scores)
        model_df_plot = model_df_plot.sort_values("score", ascending=False)

        bars = ax.barh(model_df_plot["model"], model_df_plot["score"], color=COLORS)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                ha="left",
                va="center",
                fontweight="bold",
            )

        ax.set_xlabel(f"{metric_name} (%)", fontsize=11, fontweight="bold")
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Model Performance Comparison (Aggregate Across Categories)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if output_path is None:
        output_path = PLOTS_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")
    return output_path


def plot_heatmap_category_model(df: pd.DataFrame,metric: str = "hallucination_pass_rate",output_path: Optional[Path] = None) -> Path:
    """
    Create a heatmap showing metric values across categories and models.
    """
    # Pivot for heatmap
    pivot_data = df.pivot_table(values=metric, index="category", columns="model", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(pivot_data,annot=True,fmt=".1f",cmap="RdYlGn",cbar_kws={"label": f"{metric} (%)"},ax=ax,linewidths=0.5)

    ax.set_title(
        f"{metric.replace('_', ' ').title()} Heatmap\n(Category × Model)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax.set_ylabel("Category", fontsize=11, fontweight="bold")

    plt.tight_layout()

    if output_path is None:
        metric_name = metric.replace("_", "_")
        output_path = PLOTS_DIR / f"heatmap_{metric_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")
    return output_path


def plot_metric_distributions(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Plot distributions of metric scores (not pass/fail, but continuous scores).
    """
    score_metrics = [
        ("hallucination_avg_score", "Hallucination Score"),
        ("answer_relevancy_avg_score", "Answer Relevancy Score"),
        ("faithfulness_avg_score", "Faithfulness Score"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = df["model"].unique()

    for idx, (score_col, score_name) in enumerate(score_metrics):
        ax = axes[idx]

        # Filter out NaNs
        plot_df = df[[score_col, "model"]].dropna()

        if len(plot_df) == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            continue

        # Create violin plot
        parts = ax.violinplot(  # noqa: F841
            [
                plot_df[plot_df["model"] == model][score_col].values
                for model in sorted(models)
            ],
            positions=range(len(models)),
            showmeans=True,
            showmedians=True,
        )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(sorted(models))
        ax.set_ylabel(f"{score_name}", fontsize=11, fontweight="bold")
        ax.set_title(score_name, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

    plt.suptitle("Metric Score Distributions by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is None:
        output_path = PLOTS_DIR / "metric_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")
    return output_path


def plot_sample_count_by_category(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Bar chart showing number of samples per category and model.
    """
    sample_counts = df.groupby(["category", "model"]).size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(12, 6))

    categories = sample_counts["category"].unique()
    models = sample_counts["model"].unique()
    x = np.arange(len(categories))
    width = 0.35

    for i, model in enumerate(models):
        model_data = sample_counts[sample_counts["model"] == model]
        values = []
        for cat in categories:
            cat_model = model_data[model_data["category"] == cat]
            if len(cat_model) > 0:
                values.append(cat_model["count"].values[0])
            else:
                values.append(0)

        offset = (i - len(models) / 2) * width
        ax.bar(x + offset, values, width, label=model, color=COLORS[i])

    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_title("Sample Count by Category & Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend(title="Model", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path is None:
        output_path = PLOTS_DIR / "sample_counts_by_category.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")
    return output_path


# ============================================================================
# Report Generation
# ============================================================================


def generate_comparison_report(df: pd.DataFrame, summary_stats: dict, output_path: Optional[Path] = None) -> Path:
    """
    Generate a comprehensive text-based model comparison report.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "model_comparison_report.txt"

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("TruthfulQA Evaluation using DeepEval Metrics\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        overall = summary_stats.get("overall", {})
        f.write(f"Total Categories:        {overall.get('total_categories', 'N/A')}\n")
        f.write(f"Total Models Evaluated:  {overall.get('total_models', 'N/A')}\n")
        f.write(f"Total Evaluations:       {overall.get('total_evaluations', 'N/A')}\n")
        f.write(
            f"Avg Hallucination Rate:  {overall.get('avg_hallucination_pass_rate', 0):.2f}%\n"
        )
        f.write(
            f"Avg Relevancy Rate:      {overall.get('avg_relevancy_pass_rate', 0):.2f}%\n"
        )
        f.write(
            f"Avg Faithfulness Rate:   {overall.get('avg_faithfulness_pass_rate', 0):.2f}%\n"
        )
        f.write("\n")

        # Per-model statistics
        f.write("PER-MODEL STATISTICS\n")
        f.write("-" * 80 + "\n")

        for model in sorted(df["model"].unique()):
            model_df = df[df["model"] == model]
            f.write(f"\nModel: {model}\n")
            f.write(f"  Categories Evaluated: {model_df['category'].nunique()}\n")
            f.write(f"  Total Samples:        {len(model_df)}\n")
            f.write(
                f"  Avg Hallucination Rate: "
                f"{model_df['hallucination_pass_rate'].mean():.2f}%\n"
            )
            f.write(
                f"  Avg Relevancy Rate:     "
                f"{model_df['answer_relevancy_pass_rate'].mean():.2f}%\n"
            )
            f.write(
                f"  Avg Faithfulness Rate:  "
                f"{model_df['faithfulness_pass_rate'].mean():.2f}%\n"
            )

        # Per-category statistics
        f.write("\n\nPER-CATEGORY STATISTICS\n")
        f.write("-" * 80 + "\n")

        for category in sorted(df["category"].unique()):
            cat_df = df[df["category"] == category]
            f.write(f"\n{category.upper()}\n")
            f.write(f"  Samples: {cat_df['sample_count'].sum()}\n")

            for model in sorted(cat_df["model"].unique()):
                model_cat = cat_df[cat_df["model"] == model]
                if len(model_cat) > 0:
                    f.write(f"    {model}:\n")
                    f.write(
                        f"      Hallucination Pass Rate: "
                        f"{model_cat['hallucination_pass_rate'].values[0]:.2f}%\n"
                    )
                    f.write(
                        f"      Relevancy Pass Rate:     "
                        f"{model_cat['answer_relevancy_pass_rate'].values[0]:.2f}%\n"
                    )
                    f.write(
                        f"      Faithfulness Pass Rate:  "
                        f"{model_cat['faithfulness_pass_rate'].values[0]:.2f}%\n"
                    )

        # Summary and recommendations
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("SUMMARY & RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n")

        best_model = df.groupby("model")["hallucination_pass_rate"].mean().idxmax()
        f.write(f"\n✅ Best performing model (overall hallucination): {best_model}\n")

        worst_cat = df.groupby("category")["hallucination_pass_rate"].mean().idxmin()
        f.write(f"⚠️  Weakest category (overall hallucination): {worst_cat}\n")

        f.write("\nNote: All metrics are pass rates (higher = better).\n")
        f.write("Hallucination metric: % of responses that pass (no hallucinations)\n")

    print(f"✅ Saved: {output_path}")
    return output_path


# ============================================================================
# Main Pipeline
# ============================================================================


def generate_plots_and_reports(
    csv_path: Optional[Path] = None, summary_stats_path: Optional[Path] = None
):
    """Main plotting pipeline."""
    print("\n" + "=" * 80)
    print("Plotting & Report Generation Pipeline")
    print("=" * 80 + "\n")

    if csv_path is None:
        csv_path = OUTPUT_DIR / "category_aggregated.csv"
    if summary_stats_path is None:
        summary_stats_path = OUTPUT_DIR / "category_summary_stats.json"

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return

    # Load data
    print(f"Loading aggregated data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} records\n")

    # Load summary stats
    summary_stats = {}
    if summary_stats_path.exists():
        with open(summary_stats_path) as f:
            summary_stats = json.load(f)
        print("✅ Loaded summary statistics\n")

    # Generate plots
    print("Generating plots...\n")
    plot_paths = {
        "hallucination_by_category": plot_hallucination_rate_by_category(df),
        "model_comparison": plot_model_comparison(df),
        "heatmap_hallucination": plot_heatmap_category_model(
            df, "hallucination_pass_rate"
        ),
        "heatmap_relevancy": plot_heatmap_category_model(
            df, "answer_relevancy_pass_rate"
        ),
        "heatmap_faithfulness": plot_heatmap_category_model(
            df, "faithfulness_pass_rate"
        ),
        "metric_distributions": plot_metric_distributions(df),
        "sample_counts": plot_sample_count_by_category(df),
    }

    # Generate report
    print("\nGenerating comparison report...\n")
    report_path = generate_comparison_report(df, summary_stats)

    print("=" * 80)
    print("Plotting & Report Generation Complete!")
    print("=" * 80)
    print(f"\n📊 Plots saved to: {PLOTS_DIR}/")
    print(f"📄 Report saved to: {report_path}\n")

    return plot_paths, report_path


if __name__ == "__main__":
    generate_plots_and_reports()

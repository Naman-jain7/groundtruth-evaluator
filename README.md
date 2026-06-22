# TruthfulQA Evaluation Pipeline with DeepEval

## 📋 Overview

A production-ready Python pipeline for evaluating LLM responses on the TruthfulQA dataset using DeepEval metrics. Evaluates hallucination, answer relevancy, and faithfulness across multiple models and topic categories.

**Pipeline Stages:**
1. **Output Generation** - Query Ollama Cloud models with TruthfulQA questions
2. **Category Aggregation** - Group results by topic, calculate per-category metrics
3. **Visualization & Reporting** - Generate plots and comparison reports

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- `uv` package manager
- Ollama Cloud API key

### Installation

```bash
# Clone/download the project
cd truthfulqa-evaluation

# Install dependencies with uv
uv pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OLLAMA_API_KEY
```

### Run the Full Pipeline

```bash
# Quick test with 5 sample questions
python main.py --sample 5

# Full evaluation (all questions)
python main.py

# Run specific stages
python main.py --stage generate  # Only generate outputs
python main.py --stage aggregate # Only aggregate
python main.py --stage plot      # Only plot
```

---

## 📁 Project Structure

```
.
├── main.py                      # Pipeline orchestrator
├── generate_outputs.py          # Stage 1: Query models & evaluate
├── aggregate_categories.py      # Stage 2: Group by category
├── generate_plots.py            # Stage 3: Visualize & report
├── requirements.txt             # Dependencies
├── .env.example                 # Configuration template
└── outputs/                     # Generated files
    ├── evaluation_results.csv         # Raw per-question results
    ├── category_aggregated.csv        # Category-level aggregation
    ├── category_summary_stats.json    # Summary statistics
    ├── model_comparison_report.txt    # Text report
    └── plots/
        ├── hallucination_by_category.png
        ├── model_comparison.png
        ├── heatmap_*.png
        ├── metric_distributions.png
        └── sample_counts_by_category.png
```

---

## 📊 Stages Explained

### Stage 1: Output Generation (`generate_outputs.py`)

**What it does:**
- Loads TruthfulQA dataset from Hugging Face
- Queries each question through Ollama Cloud API (Models A & B)
- Evaluates responses using three DeepEval metrics:
  - **HallucinationMetric**: Detects factually incorrect statements
  - **AnswerRelevancyMetric**: Checks if answer addresses the question
  - **FaithfulnessMetric**: Verifies consistency with ground truth

**Key Features:**
- Exponential backoff retry logic for API reliability
- Per-question evaluation scores and pass/fail status
- Detailed error handling and logging
- Progress bars with tqdm

**Input:** TruthfulQA dataset (auto-downloaded from Hugging Face)
**Output:** `evaluation_results.csv` with columns:
```
question_id, question, category, model, answer, correct_answer,
hallucination_score, hallucination_metric,
answer_relevancy_score, answer_relevancy_metric,
faithfulness_score, faithfulness_metric,
error, timestamp
```

**Usage:**
```python
from generate_outputs import generate_outputs

# Sample 10 questions from each category
results, csv_path = generate_outputs(sample_questions=10)
```

---

### Stage 2: Category Aggregation (`aggregate_categories.py`)

**What it does:**
- Groups results by topic category (health, law, finance, etc.)
- Calculates aggregate metrics per category-model pair
- Computes hallucination pass rates (% of non-hallucinating responses)
- Generates summary statistics

**Output Files:**
1. **category_aggregated.csv** - Per-category metrics:
```
category, model, sample_count,
hallucination_pass_rate, hallucination_avg_score,
answer_relevancy_pass_rate, answer_relevancy_avg_score,
faithfulness_pass_rate, faithfulness_avg_score
```

2. **category_aggregated.json** - Structured category data

3. **category_summary_stats.json** - High-level statistics:
   - Models evaluated per category
   - Overall performance metrics
   - Total samples processed

**Hallucination Rate Calculation:**
```
hallucination_pass_rate = (# of responses passing hallucination check / total) × 100%

Higher rate = Fewer hallucinations = Better performance
```

**Usage:**
```python
from aggregate_categories import aggregate_categories

results = aggregate_categories(csv_path="outputs/evaluation_results.csv")
# Access: results['aggregated'][category_name]
# Access: results['hallucination_rates'][category_model_key]
```

---

### Stage 3: Plotting & Reporting (`generate_plots.py`)

**Visualizations Generated:**

1. **hallucination_by_category.png**
   - Bar chart: Hallucination pass rate by category & model
   - Higher bars = Better (fewer hallucinations)

2. **model_comparison.png**
   - 3-panel comparison: Hallucination, Relevancy, Faithfulness
   - Aggregate metrics across all categories

3. **heatmap_hallucination_pass_rate.png**
   - Color-coded matrix: Categories (rows) × Models (columns)
   - Red = Low performance, Green = High performance

4. **heatmap_answer_relevancy_pass_rate.png** & **heatmap_faithfulness_pass_rate.png**
   - Similar heatmaps for other metrics

5. **metric_distributions.png**
   - Violin plots: Score distributions per model
   - Shows variance and central tendency

6. **sample_counts_by_category.png**
   - Bar chart: Number of evaluation samples per category

**Text Report: `model_comparison_report.txt`**
```
OVERALL STATISTICS
  Total Categories: 13
  Total Models: 2
  Avg Hallucination Rate: 72.5%
  ...

PER-MODEL STATISTICS
  Model: mistral
    Avg Hallucination Rate: 75.3%
    ...

PER-CATEGORY STATISTICS
  HEALTH
    mistral: 78.2% hallucination pass rate
    neural-chat: 71.5% hallucination pass rate
  ...
```

**Usage:**
```python
from generate_plots import generate_plots_and_reports

plots, report = generate_plots_and_reports(
    csv_path="outputs/category_aggregated.csv",
    summary_stats_path="outputs/category_summary_stats.json"
)
```

---

## ⚙️ Configuration

### `.env` File

```bash
# Required
OLLAMA_API_KEY=your_key_here
OLLAMA_API_URL=https://api.ollama.ai/v1

# Models to evaluate
MODEL_A=mistral
MODEL_B=neural-chat

# Optional
REQUEST_TIMEOUT=60
BATCH_SIZE=10
```

### Command-Line Arguments

```bash
# Limit questions for testing
python main.py --sample 10 --limit 100

# Skip output generation (use existing CSV)
python main.py --skip-generation --stage plot

# Run only specific stage
python main.py --stage generate
python main.py --stage aggregate
python main.py --stage plot
```

---

## 📈 Understanding the Metrics

### HallucinationMetric
- **What:** Detects factually incorrect or made-up information
- **Score:** 0-1 (higher is better)
- **Pass Rate:** % of responses flagged as non-hallucinating
- **Use Case:** Critical for trustworthiness

### AnswerRelevancyMetric
- **What:** Checks if answer addresses the question
- **Score:** 0-1 (higher is better)
- **Pass Rate:** % of responses deemed relevant
- **Use Case:** Measure task alignment

### FaithfulnessMetric
- **What:** Verifies consistency with ground truth context
- **Score:** 0-1 (higher is better)
- **Pass Rate:** % of responses consistent with facts
- **Use Case:** Measure factual accuracy

---

## 🔧 Troubleshooting

### API Connection Issues
```
Error: "Timeout on mistral (attempt 1/3). Retrying in 2s..."
```
- Check `OLLAMA_API_KEY` is valid
- Verify `OLLAMA_API_URL` is accessible
- Increase `REQUEST_TIMEOUT` in `.env`

### DeepEval Metric Errors
```
⚠️ Hallucination metric error: [specific error]
```
- Ensure dataset is properly loaded
- Check ground truth answers are non-empty
- Verify DeepEval version compatibility

### Memory Issues with Large Datasets
```bash
# Process in batches
python main.py --sample 50  # Test with 50 questions
python main.py --limit 500  # Limit dataset to 500
```

### No Output Directory
```bash
mkdir -p outputs
```

---

## 📊 Example Output Interpretation

### CSV Result Row
```csv
question_id,question,category,model,answer,correct_answer,hallucination_score,hallucination_metric,...
gen_0001,"What is the capital of France?",geography,"mistral","Paris is the capital of France.","Paris",0.95,True,...
```
- `hallucination_metric=True` → No hallucinations detected ✅
- `hallucination_score=0.95` → Very confident in faithfulness (0-1 scale)

### Category Aggregation
```csv
category,model,sample_count,hallucination_pass_rate,answer_relevancy_pass_rate,faithfulness_pass_rate
health,mistral,45,78.5,82.3,81.2
health,neural-chat,45,71.2,79.5,75.8
```
- `mistral` performs better on health questions
- ~78.5% of responses are non-hallucinating

### Report Summary
```
Best performing model: mistral
Weakest category: finance
```

---

## 🔄 Advanced Usage

### Processing Custom Dataset
```python
from generate_outputs import load_truthful_qa

# Load with custom filters
data = load_truthful_qa(split="generation", limit=1000)

# Filter by category
health_questions = [q for q in data if q['category'] == 'health']
```

### Running Individual Pipeline Stages
```python
# Stage 1 only
from generate_outputs import generate_outputs
results, csv_path = generate_outputs(sample_questions=10)

# Stage 2 only
from aggregate_categories import aggregate_categories
agg = aggregate_categories(csv_path)

# Stage 3 only
from generate_plots import generate_plots_and_reports
plots, report = generate_plots_and_reports()
```

### Custom Metrics Evaluation
```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric

test_case = LLMTestCase(
    input="What is 2+2?",
    actual_output="2+2 equals 4",
    context=["2+2 equals 4"]
)

metric = HallucinationMetric(threshold=0.5)
metric.measure(test_case)
print(f"Score: {metric.score}, Pass: {metric.is_successful()}")
```

---

## 📝 Logging & Debugging

### Enable Verbose Output
```bash
# Add debugging to generate_outputs.py
python main.py --sample 5 -v
```

### Check Individual Question Results
```bash
# Query a specific question ID from CSV
grep "gen_0001" outputs/evaluation_results.csv
```

### View Category Breakdown
```bash
# Show hallucination rates by category
grep "health\|law\|finance" outputs/category_aggregated.csv | cut -d, -f1,2,4
```

---

## 🎯 Best Practices

1. **Testing First**
   ```bash
   # Start with 5-10 samples to verify setup
   python main.py --sample 5
   ```

2. **Monitor API Costs**
   - Ollama Cloud pricing varies by model
   - Use `--sample` flag to test before full run

3. **Handle Failures Gracefully**
   - Pipeline auto-retries failed API calls
   - Failed evaluations are logged in CSV with `error` field
   - Resume from existing CSV with `--skip-generation`

4. **Analyze Results Incrementally**
   ```bash
   # Generate outputs
   python main.py --stage generate --sample 100
   
   # Review CSV before continuing
   head -20 outputs/evaluation_results.csv
   
   # Aggregate if satisfied
   python main.py --stage aggregate --skip-generation
   ```

---

## 📚 References

- **DeepEval Documentation:** https://docs.confident-ai.com/
- **TruthfulQA Dataset:** https://huggingface.co/datasets/truthful_qa
- **Ollama Cloud API:** https://ollama.ai/api
- **Hugging Face Datasets:** https://huggingface.co/docs/datasets/

---

## 📄 License

[Your License Here]

---

## 🤝 Contributing

[Contribution Guidelines]

---

## ⚠️ Known Limitations

1. **API Rate Limits:** Ollama Cloud may rate-limit requests. Use exponential backoff.
2. **Context Length:** Very long questions may be truncated by API.
3. **Model Availability:** Some models may not be available in your region.
4. **DeepEval Thresholds:** Default 0.5 threshold may need tuning per use case.

---

## 🚀 Future Enhancements

- [ ] Support for custom models via local Ollama
- [ ] Streaming response support for faster evaluation
- [ ] Interactive dashboard for result exploration
- [ ] Export reports to PDF/HTML
- [ ] Statistical significance testing
- [ ] Per-category performance recommendations
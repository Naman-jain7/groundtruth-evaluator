
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from config import AVAILABLE_MODELS

load_dotenv()

from aggregate_categories import aggregate_categories  # noqa: E402
from config import OUTPUT_DIR  # noqa: E402
from generate_answers import generate_answers  # noqa: E402
from evaluate_answers import evaluate_answers  # noqa: E402
from generate_plots import generate_plots_and_reports  # noqa: E402
from main import run_full_pipeline  # noqa: E402


st.set_page_config(page_title="Ground Truth Evaluator", layout="wide")
st.title("Ground Truth Evaluator Dashboard")

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")

st.sidebar.subheader("Model Selection")
model_1 = st.sidebar.selectbox("Model 1", options=[AVAILABLE_MODELS[0], AVAILABLE_MODELS[1]], index=0)
model_2 = st.sidebar.selectbox("Model 2", options=[AVAILABLE_MODELS[0], AVAILABLE_MODELS[1]], index=1)
eval_model = st.sidebar.selectbox("Evaluation Judge Model", options=[AVAILABLE_MODELS[2]], index=0)

st.sidebar.subheader("Pipeline Options")
sample_questions = st.sidebar.number_input("Sample Questions (0 = all)", min_value=0, value=5, step=1)
dataset_limit = st.sidebar.number_input("Dataset Limit (0 = None)", min_value=0, value=0, step=1)

stage = st.sidebar.selectbox(
    "Execution Stage",
    options=["all", "answers", "evaluate", "aggregate", "plot"],
    index=0,
    help="answers: generate raw answers | evaluate: run DeepEval metrics | aggregate: group by category | plot: generate charts"
)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
st.markdown("### Run Pipeline")
st.write("Configure your settings in the sidebar, then click the button below to start.")

if st.button("Run Evaluator", type="primary"):
    # Convert limit 0 to None for the pipeline functions
    actual_limit = dataset_limit if dataset_limit > 0 else None
    
    # We will capture output in a placeholder
    status_placeholder = st.empty()
    status_placeholder.info("Running pipeline... Please wait.")
    
    try:
        if stage == "all":
            success = run_full_pipeline(
                eval_model_name=eval_model, # type: ignore
                models=[model_1, model_2],
                dataset_limit=actual_limit,
                sample_questions=sample_questions,
            )
            if success:
                status_placeholder.success("Pipeline completed successfully!")
            else:
                status_placeholder.error("Pipeline failed. Check terminal logs.")

        elif stage == "answers":
            results, raw_path = generate_answers(
                models=[model_1, model_2],
                dataset_limit=actual_limit,
                sample_questions=sample_questions,
            )
            if raw_path:
                status_placeholder.success(f"Answers generated! Saved to {raw_path}")
            else:
                status_placeholder.error("Answer generation failed. Check terminal logs.")

        elif stage == "evaluate":
            eval_results, eval_path = evaluate_answers(
                eval_model_name=eval_model, # type: ignore
            )
            if eval_path:
                status_placeholder.success(f"Evaluation completed! Saved to {eval_path}")
            else:
                status_placeholder.error("Evaluation failed. Check terminal logs.")

        elif stage == "aggregate":
            evaluated_path = OUTPUT_DIR / "evaluated_results.csv"
            aggregate_categories(evaluated_path)
            status_placeholder.success("Aggregation completed!")

        elif stage == "plot":
            plot_paths, report_path = generate_plots_and_reports() # type: ignore
            status_placeholder.success(f"Plotting completed! Report saved to {report_path}")

    except Exception as e:
        status_placeholder.error(f"An error occurred: {str(e)}")





# Display Results if they exist
st.markdown("### Latest Results")

evaluated_path = OUTPUT_DIR / "evaluated_results.csv"
raw_path = OUTPUT_DIR / "raw_answers.csv"

if evaluated_path.exists():
    try:
        df = pd.read_csv(evaluated_path)
        st.dataframe(df.tail(10))  # Show last 10 rows
    except Exception:
        pass
elif raw_path.exists():
    st.info("Raw answers found but not yet evaluated. Run the 'evaluate' stage to score them.")
    try:
        df = pd.read_csv(raw_path)
        st.dataframe(df.tail(10))
    except Exception:
        pass
else:
    st.write("No results found yet. Run the pipeline to generate them.")


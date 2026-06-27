
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from aggregate_categories import aggregate_categories  # noqa: E402
from config import OUTPUT_DIR  # noqa: E402
from generate_outputs import generate_outputs  # noqa: E402
from generate_plots import generate_plots_and_reports  # noqa: E402
from main import run_full_pipeline  # noqa: E402

st.set_page_config(page_title="Ground Truth Evaluator", layout="wide")
st.title("Ground Truth Evaluator Dashboard")

AVAILABLE_MODELS = {"gemma3:4b", "gemma4:31b", "gpt-oss:120b"}

available_models = sorted(list(AVAILABLE_MODELS))

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")

st.sidebar.subheader("Model Selection")
model_1 = st.sidebar.selectbox("Model 1", options=available_models, index=0)
model_2 = st.sidebar.selectbox("Model 2", options=available_models, index=1)
eval_model = st.sidebar.selectbox("Evaluation Judge Model", options=available_models, index=2)

st.sidebar.subheader("Pipeline Options")
sample_questions = st.sidebar.number_input("Sample Questions (0 = all)", min_value=0, value=5, step=1)
dataset_limit = st.sidebar.number_input("Dataset Limit (0 = None)", min_value=0, value=0, step=1)
skip_generation = st.sidebar.checkbox("Skip Generation (use existing CSV)", value=False)

stage = st.sidebar.selectbox(
    "Execution Stage",
    options=["all", "generate", "aggregate", "plot"],
    index=0
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
                eval_model_name=eval_model,
                models=[model_1, model_2],
                dataset_limit=actual_limit,
                sample_questions=sample_questions,
                skip_generation=skip_generation
            )
            if success:
                status_placeholder.success("Pipeline completed successfully!")
            else:
                status_placeholder.error("Pipeline failed. Check terminal logs.")
                
        elif stage == "generate":
            results, csv_path = generate_outputs(
                eval_model_name=eval_model,
                models=[model_1, model_2],
                dataset_limit=actual_limit,
                sample_questions=sample_questions
            ) # type: ignore
            status_placeholder.success(f"Generation completed! Results saved to {csv_path}")
            
        elif stage == "aggregate":
            aggregate_categories()
            status_placeholder.success("Aggregation completed!")
            
        elif stage == "plot":
            plot_paths, report_path = generate_plots_and_reports() # type: ignore
            status_placeholder.success(f"Plotting completed! Report saved to {report_path}")
            
    except Exception as e:
        status_placeholder.error(f"An error occurred: {str(e)}")





# Display Results if they exist
st.markdown("### Latest Results")

csv_path = OUTPUT_DIR / "evaluation_results.csv"
if csv_path.exists():
    try:
        df = pd.read_csv(csv_path)
        st.dataframe(df.tail(10)) # Show last 10 rows
    except Exception:
        pass
else:
    st.write("No evaluation results found yet. Run the pipeline to generate them.")

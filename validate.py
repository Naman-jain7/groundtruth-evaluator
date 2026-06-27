"""
Testing and validation utilities for the TruthfulQA evaluation pipeline.
Includes: API connectivity tests, data validation, metric verification.
"""

import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv


def test_environment():
    """Verify environment setup and dependencies."""
    print("\n" + "=" * 80)
    print("ENVIRONMENT VALIDATION TEST")
    print("=" * 80 + "\n")

    # Check Python version
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(
            f"  ✅ Python {version.major}.{version.minor}.{version.micro} (required: 3.11+)\n"
        )
    else:
        print(f"  ❌ Python {version.major}.{version.minor} (required: 3.11+)\n")
        return False

    # Check .env file
    print("✓ Checking .env configuration...")
    env_path = Path(".env")
    if not env_path.exists():
        print(f"  ❌ .env file not found. Create with: cp .env.example .env\n")
        return False
    print(f"  ✅ .env file found\n")

    # Load environment
    load_dotenv()

    # Check API key
    print("✓ Checking Ollama API key...")
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print(f"  ❌ OLLAMA_API_KEY not set or is placeholder\n")
        return False
    print(f"  ✅ API key configured (length: {len(api_key)})\n")

    # Check API URL
    print("✓ Checking Ollama API URL...")
    api_url = os.getenv("OLLAMA_API_URL", "https://api.ollama.ai/v1")
    print(f"  ✅ API URL: {api_url}\n")

    # Check models
    print("✓ Checking model configuration...")
    model_a = os.getenv("MODEL_A", "mistral")
    model_b = os.getenv("MODEL_B", "neural-chat")
    print(f"  ✅ Model A: {model_a}")
    print(f"  ✅ Model B: {model_b}\n")

    return True


def test_dependencies():
    """Verify required Python packages are installed."""
    print("=" * 80)
    print("DEPENDENCY VALIDATION TEST")
    print("=" * 80 + "\n")

    required_packages = {
        "dotenv": "python-dotenv",
        "requests": "requests",
        "datasets": "datasets",
        "pandas": "pandas",
        "numpy": "numpy",
        "deepeval": "deepeval",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
    }

    all_ok = True
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {module:20s} {package}")
        except ImportError:
            print(f"❌ {module:20s} {package} - NOT INSTALLED")
            all_ok = False

    if not all_ok:
        print(
            f"\n⚠️  Install missing packages with: uv pip install -r requirements.txt\n"
        )
        return False

    print("\n✅ All dependencies installed\n")
    return True


def test_api_connection():
    """Test connection to Ollama Cloud API."""
    print("=" * 80)
    print("API CONNECTION TEST")
    print("=" * 80 + "\n")

    try:
        import requests
        from ollama_client import OllamaCloudClient

        print("Initializing Ollama Cloud client...")
        client = OllamaCloudClient()
        print("✅ Client initialized\n")

        # Test simple generation
        model = os.getenv("MODEL_A", "mistral")
        print(f"Testing API call with model: {model}")
        print("Prompt: 'What is 2+2?'\n")

        response = client.generate(
            model=model,
            prompt="What is 2+2?",
            max_tokens=50,
            retries=2,
        )

        print(f"✅ API Response received:")
        print(f"   {response}\n")
        return True

    except Exception as e:
        print(f"❌ API connection failed: {e}\n")
        return False


def test_datasets():
    """Test TruthfulQA dataset loading."""
    print("=" * 80)
    print("DATASET VALIDATION TEST")
    print("=" * 80 + "\n")

    try:
        from datasets import load_from_disk

        print("Loading TruthfulQA from disk...")
        dataset = load_from_disk("data/truthful_qa")

        # Show first 3 items
        print(f"✅ Dataset loaded successfully")
        print(f"   Total samples: {len(dataset)}\n")

        print("Sample structure (first item):")
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, list):
                print(f"   {key}: [{len(value)} items]")
            elif isinstance(value, str) and len(str(value)) > 60:
                print(f"   {key}: {str(value)[:60]}...")
            else:
                print(f"   {key}: {value}")

        print("\n✅ Dataset structure valid\n")
        return True

    except Exception as e:
        print(f"❌ Dataset loading failed: {e}\n")
        return False


def test_deepeval_metrics():
    """Test DeepEval metrics initialization."""
    print("=" * 80)
    print("DEEPEVAL METRICS VALIDATION TEST")
    print("=" * 80 + "\n")

    try:
        from evaluator import DeepEvalEvaluator

        print("Initializing evaluator...")
        evaluator = DeepEvalEvaluator()
        print("✅ Evaluator initialized\n")

        print("Creating test case...")
        question = "What is the capital of France?"
        answer = "Paris is the capital of France."
        context = "Paris is the capital of France"
        print("✅ Test case created\n")

        print("Evaluating test case...")
        eval_results = evaluator.evaluate(question, answer, context)

        print(f"✅ Metrics evaluated successfully:")
        print(
            f"   Hallucination Score:  {eval_results.get('hallucination_score')} (pass: {eval_results.get('hallucination_pass')})"
        )
        print(
            f"   Relevancy Score:      {eval_results.get('answer_relevancy_score')} (pass: {eval_results.get('answer_relevancy_metric')})"
        )
        print(
            f"   Faithfulness Score:   {eval_results.get('faithfulness_score')} (pass: {eval_results.get('faithfulness_metric')})\n"
        )

        return True

    except Exception as e:
        print(f"❌ Metrics evaluation failed: {e}\n")
        return False


def test_output_directory():
    """Verify output directory is writable."""
    print("=" * 80)
    print("OUTPUT DIRECTORY VALIDATION TEST")
    print("=" * 80 + "\n")

    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        print(f"✅ Output directory exists: {output_dir.absolute()}\n")

        # Test write permission
        test_file = output_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        print(f"✅ Output directory is writable\n")

        return True

    except Exception as e:
        print(f"❌ Output directory test failed: {e}\n")
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "🧪 " * 40)
    print("\n" + "=" * 80)
    print("TRUTHFULQA PIPELINE - VALIDATION TEST SUITE")
    print("=" * 80)

    tests = [
        ("Environment Setup", test_environment),
        ("Dependencies", test_dependencies),
        ("Output Directory", test_output_directory),
        ("TruthfulQA Dataset", test_datasets),
        ("DeepEval Metrics", test_deepeval_metrics),
        ("Ollama Cloud API", test_api_connection),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}\n")
            results[test_name] = False

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80 + "\n")

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready to run pipeline!")
        print("=" * 80 + "\n")
        print("Next steps:")
        print("  python main.py --sample 5      # Test with 5 questions")
        print("  python main.py                  # Run full pipeline")
        print("\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED - See details above")
        print("=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

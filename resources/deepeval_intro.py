import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq_model import GroqModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate
from dotenv import load_dotenv

load_dotenv()


# -------------------------
# Groq judge
# -------------------------

groq_model = GroqModel(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# -------------------------
# Test cases
# -------------------------

case_1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)

case_2 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="France is a beautiful country famous for its food and wine.",
)


# -------------------------
# Metric
# -------------------------

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=groq_model,
    include_reason=True,
)


# -------------------------
# Evaluate
# -------------------------

evaluate(
    test_cases=[case_1, case_2],
    metrics=[metric],
)
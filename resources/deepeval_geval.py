import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq_model import GroqModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import evaluate


JUDGE_MODEL = GroqModel(
    model="openai/gpt-oss-20b",
    temperature=0,
)
THRESHOLD = 0.7

correctness = GEval(
    name="correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    model=JUDGE_MODEL,
    threshold=THRESHOLD
)

test_case = LLMTestCase(
    input="what is offline eval?",
    actual_output="Offline eval runs before deployment.",
    expected_output="Offline eval is evaluation done before deployment using a fixed golden dataset."
)

evaluate(test_cases=[test_case], metrics=[correctness])
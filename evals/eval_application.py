
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq_model import GroqModel

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import evaluate
from deepeval.metrics.g_eval import Rubric

from src.rag_pipeline import RagPipeline

from dotenv import load_dotenv
load_dotenv()


GOLDEN_PATH = "goldens/correctness_goldens.json"    # question + ideal_answer
JUDGE_MODEL = GroqModel(
    model="openai/gpt-oss-20b",
    temperature=0,
)
THRESHOLD = 0.7


# 1. LOAD queries + ideal_answers (ideal_answer is the CORRECT answer, our reference
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)
goldens = goldens[:3]


# 2. RUN FULL PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g["question"])                # retrieve -> rerank -> generate

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            actual_output=result["answer"],        # what the GENERATOR produced
            expected_output=g["ideal_answer"]    # the CORRECT reference answer
        )
    )

# 3. THE CORRECTNESS METRIC (graded G-Eval - partial credit, not pass/fail)
correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare the actual output against the key facts in the corrected output.",
        "Heavily penalize statements in the actual output that contradict the expected output or are factually wrong.",
        "Reward statements that match the expected output in meaning, regardless of wording.",
        "Do NOT penalize the actual output for omitting information - only wrong statements count here."
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    strict_mode=False          # graded_scale: strict_mode=True would collapse it to 0/1
)


# EVALUATE
evaluate(test_cases=test_cases, metrics=[correctness])
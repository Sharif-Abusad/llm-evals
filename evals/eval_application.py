
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
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
            "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
            Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
            Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

# EVALUATE
evaluate(test_cases=test_cases, metrics=[correctness])
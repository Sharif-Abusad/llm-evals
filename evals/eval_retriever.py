import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq_model import GroqModel

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval import evaluate

from src.retriever import build_retriever

from dotenv import load_dotenv
load_dotenv()

JUDGE_MODEL = GroqModel(
    model="openai/gpt-oss-120b",
    temperature=0,
)
GOLDEN_PATH = "goldens/retriever_goldens.json"
THRESHOLD = 0.7

# 1. Load the golden set --- the fix human-authored truth
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)
# goldens = goldens[:1]

# 2. Run the RETRIEVER on each question to fill retrieval_context,
#    then build one test case per golden.
retriever = build_retriever()

test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content[:500] for doc in retrieved[:2]]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not eval in this run)",
        )
    )


# 3. THE METRICS --- recall(did we miss?) and precision (ranked well?)
metrics = [
    ContextualRecallMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=False,
        async_mode=False,
    ),

    ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=False,
        async_mode=False,
    ),
]


# 4. EVALUATE --- every metric on every case batched + parallel, with a parallel report
# evaluate(
#     test_cases=test_cases,
#     metrics=metrics,
#     hyperparameters={
#         "retriever": "base_k5",
#         "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
#         "chunk_size": 750,
#         "chunk_overlap": 100,
#         "top_k": 3,
#         "judge_model": "openai/gpt-oss-120b",
#         "golden_set": GOLDEN_PATH,
#     },
# )
# ---------------------------------------------------------
# EVALUATE ONE TEST CASE AT A TIME
# ---------------------------------------------------------
import time

for i, test_case in enumerate(test_cases, start=1):

    print(f"\nEvaluating test case {i}/{len(test_cases)}")

    evaluate(
        test_cases=[test_case],
        metrics=metrics,
    )

    if i < len(test_cases):
        print("Waiting 10 seconds...")
        time.sleep(10)
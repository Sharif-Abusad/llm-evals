"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

    python -m evals.eval_generator
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq_model import GroqModel

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval import evaluate
from langchain_groq import ChatGroq
from src.generator import generate

from dotenv import load_dotenv
load_dotenv()


GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = GroqModel(
    model="openai/gpt-oss-20b",
    temperature=0,
)
THRESHOLD = 0.7


# 1. LOAD the faithfulness golden set (query + ideal_context)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)
goldens = goldens[:3]

# 2. RUN GENERATOR on the GOLDEN context (isolation), build one test case each
test_cases = []
for g in goldens:
    context = g["ideal_context"]            # known-good context (list of chunk strings)
    query = g['query']                      # RUN the generator -> actual_output
    answer = generate(query, context)

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,           # the generated answer we are judging     
            retrieval_context=context,      # faithfulness checks the answer against this
            # no expected_output - faithfulness never reads it
        )
    )


# 3. THE METRIC - decomposes actual_output into claims, attributes each to context
metrics = [
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=False,     # prints WHY each score - show which claims were unsupported
        async_mode=False
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=False,
        async_mode=False
    )
]

from deepeval.evaluate.configs import AsyncConfig
# 4. EVALUATE - runs the metric on every test_case, prints a report
evaluate(
    test_cases=test_cases, 
    metrics=metrics, 
    async_config=AsyncConfig(
        run_async=False
    )
)
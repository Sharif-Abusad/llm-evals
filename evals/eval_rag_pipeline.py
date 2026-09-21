import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq_model import GroqModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from src.rag_pipeline import RagPipeline

from dotenv import load_dotenv
load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = GroqModel(
    model="openai/gpt-oss-20b",
    temperature=0,
)
THRESHOLD = 0.7


# 1. LOAD queries (we only need the queries - context comes from the pipeline now)
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)
goldens = goldens[:2]


# 2. RUN FULL PIPELINE per query, build a test case from LIVE output
rag = RagPipeline()
test_cases = []
for g in goldens:
    result = rag.invoke(g["query"])                # retrieve -> rerank -> generate

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=result["answer"],        # what the GENERATOR produced
            retrieval_context=result["context"]    # what the RETRIEVER returned
        )
    )


# 3. THREE TRIAD METRICS
metrics = [
    ContextualRelevancyMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=False,
        async_mode=False
    ),
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=False,
        async_mode=False
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=False,
        async_mode=False
    )
]


# 4. EVALUATE
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        run_async=False
    )
)
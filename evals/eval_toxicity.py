import json
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from groq_model import GroqModel

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric
from deepeval import evaluate

from src.rag_pipeline import RagPipeline

from dotenv import load_dotenv
load_dotenv()


GOLDEN_PATH = "goldens/toxicity_goldens.json"
JUDGE_MODEL = GroqModel(
    model="openai/gpt-oss-20b",
    temperature=0
)
THRESHOLD = 0.7

# 1. LOAD toxicity inputs
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)
goldens = goldens[:2]


# 2. RUN THE FULL PIPELINE per input, build a test case from LIVE output
rag = RagPipeline()
test_cases = []

for g in goldens:
    result = rag.invoke(g["input"])             # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["input"],
            actual_output=result["answer"]
        )
    )



# 3. TOXICITY — built-in DeepEval metric
#    Lower score is better. A test passes when toxicity <= threshold.
toxicity = ToxicityMetric(
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    include_reason=True,
    strict_mode=False
)


# 4. EVALUATE
evaluate(test_cases=test_cases, metrics=[toxicity])
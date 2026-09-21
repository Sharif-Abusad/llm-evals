import os
import json
import time
from typing import Optional, Type

from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM


class GroqModel(DeepEvalBaseLLM):

    def __init__(
        self,
        model="openai/gpt-oss-20b",
        temperature=0,
    ):
        self.model_name = model

        self.model = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def load_model(self):
        return self.model

    def _parse_response(self, response, schema: Optional[Type] = None):
        text = response.content.strip()

        if schema is None:
            return text

        if not text:
            raise ValueError(
                "Groq returned an empty response while DeepEval "
                "expected structured JSON."
            )

        if text.startswith("```"):
            text = text.replace("```json", "", 1)
            text = text.replace("```", "", 1)
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Groq returned invalid JSON.\n"
                f"Raw response: {text!r}"
            ) from e

        return schema.model_validate(data)

    def generate(
        self,
        prompt: str,
        schema: Optional[Type] = None,
    ):
        # Small delay to avoid hitting Groq TPM limits
        time.sleep(3)

        response = self.model.invoke(prompt)

        return self._parse_response(response, schema)

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type] = None,
    ):
        import asyncio

        # Small delay to avoid hitting Groq TPM limits
        await asyncio.sleep(3)

        response = await self.model.ainvoke(prompt)

        return self._parse_response(response, schema)

    def get_model_name(self):
        return self.model_name
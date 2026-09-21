import os
from typing import Optional, Type

from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM


class GroqModel(DeepEvalBaseLLM):

    def __init__(
        self,
        model="openai/gpt-oss-120b",
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

    def generate(
        self,
        prompt: str,
        schema: Optional[Type] = None,
    ):
        response = self.model.invoke(prompt)

        if schema:
            # Parse JSON into the Pydantic schema
            import json

            text = response.content.strip()

            if text.startswith("```"):
                text = text.replace("```json", "")
                text = text.replace("```", "")
                text = text.strip()

            data = json.loads(text)

            return schema.model_validate(data)

        return response.content

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type] = None,
    ):
        response = await self.model.ainvoke(prompt)

        if schema:
            import json

            text = response.content.strip()

            if text.startswith("```"):
                text = text.replace("```json", "")
                text = text.replace("```", "")
                text = text.strip()

            data = json.loads(text)

            return schema.model_validate(data)

        return response.content

    def get_model_name(self):
        return self.model_name
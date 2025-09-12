import asyncio
import math
import random

from dotenv import load_dotenv
from google import genai
from google.genai import types

from llm_call import LLM


SUPPORTED_MODEL = "gemini-2.5-flash"
RANKED_LIST_SYS_PROMPT = 'You are an assistant that returns a JSON object where the keys are the brand names and the values are the order you return them. Example output: {"foo": 1, "bar": 2}'
CHOICE_SYS_PROMPT = 'In one word answer the following question, strictly with the choice between two options'

load_dotenv()

class Model(LLM):
    def __init__(self):
        self.__client = genai.Client(http_options=types.HttpOptions(api_version="v1"))
    
    @property
    def computed_model_name(self) -> str:
        return SUPPORTED_MODEL
    
    def parallelism(self) -> int:
        return 2

    async def ask_for_list(self, choices: int, question: str, safe_answer: str, temperature: float | None) -> LLM.Response:
        raise NotImplementedError()

    async def conversation(self, questions: list[str], temperature: float | None) -> LLM.Conversation:
        raise NotImplementedError()

    async def ask_generic_question(self, system_prompt: str, question: str, temperature: float, is_json: bool) -> LLM.SimpleResponse:
        raise NotImplementedError()
    
    async def ask_for_open_list(self, system_prompt: str, question: str, temperature: float) -> LLM.Response:
        result = await self.ask_generic_question(system_prompt, question, temperature, is_json=True)
        return self.Response(
            answers=self.parse_json_ranked_list(result.answer),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens
        )

    async def ask_for_ranked_list(self, system_prompt: str, question: str, temperature: float) -> LLM.Response:
        return await self.ask_for_open_list(system_prompt, question, temperature)

    async def ask_generic_question_with_retries(self, system_prompt: str, question: str, temperature: float,
                                                is_json: bool, max_retries: int = 10):
        for retry in range(0, max_retries):
            try:
                return await self.ask_generic_question(system_prompt, question, temperature, is_json)
            except Exception as ex:
                if retry == max_retries - 1:
                    print(f'No more retries for {self.computed_model_name}: {ex}')
                    raise ex

                wait = pow(2, retry)
                print(f"Waiting {wait} seconds for {self.computed_model_name} to avoid {ex}")
                await asyncio.sleep(wait)

    async def choice_from_pair(self, question: str, temperature: float, max_iterations: int, system_prompt = None):
        system_prompt = "In one word answer the following question"
        return await self.ask_generic_question_with_retries(system_prompt, question, temperature, False)

    @staticmethod
    def known_models() -> set[str]:
        return {SUPPORTED_MODEL}

    @staticmethod
    def report_models() -> list[str]:
        return [SUPPORTED_MODEL]

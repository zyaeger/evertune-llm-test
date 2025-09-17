import asyncio
import math
import random
from typing import Any

import together
from together import AsyncTogether
from dotenv import dotenv_values

from constants import (
    EMPTY_ANSWER,
    EMPTY_LIST,
    CHOICE_SYS_PROMPT,
    RANKED_LIST_SYS_PROMPT,
    Choices,
)
from llm_call import LLM

SUPPORTED_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
SUPPORTED_MODEL_INTERNAL_NAME = "llama-3.1-70B"
SUPPORTED_MODEL_DISPLAY_NAME = "Llama-3.1"


class Model(LLM):
    def __init__(self):
        self.__client = AsyncTogether(
            api_key=dotenv_values("./tests/.secrets")["TOGETHER_API_KEY"]
        )

    @staticmethod
    def list_models() -> list[str]:
        return [SUPPORTED_MODEL_INTERNAL_NAME]

    @property
    def computed_model_name(self):
        return SUPPORTED_MODEL_INTERNAL_NAME

    @staticmethod
    def known_models() -> set[str]:
        return {SUPPORTED_MODEL_INTERNAL_NAME}

    @property
    def parallelism(self):
        return 100

    @property
    def has_logprob(self):
        return True

    # pylint: disable=broad-exception-caught
    async def ask_generic_question(
        self,
        system_prompt: str,
        question: str,
        temperature: float,
        is_json: bool,
    ) -> LLM.SimpleResponse:
        logprobs = 1 if not is_json else 0
        for retry in range(0, 10):
            try:
                if not is_json:
                    response = await self.__client.chat.completions.create(
                        model=SUPPORTED_MODEL,
                        messages=[
                            {"role": "user", "content": question},
                            {"role": "system", "content": system_prompt},
                        ],
                        logprobs=logprobs,
                        temperature=temperature,
                    )
                else:
                    response = await self.__client.chat.completions.create(
                        model=SUPPORTED_MODEL,
                        messages=[
                            {"role": "user", "content": question},
                            {"role": "system", "content": system_prompt},
                        ],
                        logprobs=logprobs,
                        temperature=temperature,
                        response_format={
                            "type": "json_object",
                            "schema": Choices.model_json_schema(),
                        },
                    )

                # pylint: disable=fixme
                return LLM.SimpleResponse(
                    answer=response.choices[0].message.content,
                    probability=self.extract_logprobs(response),
                    input_tokens=0,  # TODO
                    output_tokens=0,  # TODO
                )
            except together.error.TogetherException as ex:
                if ex.http_status in {429, 502, 503}:
                    seconds = int(
                        pow(2, retry) * (1.0 + random.randint(0, 100) / 100.0)
                    )
                    print(
                        f"Llama on Together {ex.http_status}: waiting {seconds} seconds to avoid quota error attempt {retry}."
                    )
                    await asyncio.sleep(seconds)
                else:
                    print(f"Error in Llama on Together: {ex}")
                    return EMPTY_ANSWER
            except Exception as ex:
                print(f"Error in Llama on Together: {ex}")
                return EMPTY_ANSWER

    # pylint: disable=broad-exception-caught
    async def ask_for_open_list(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.Response:
        response = await self.ask_generic_question(
            system_prompt, question, temperature, True
        )
        try:
            answers = self.parse_json_ranked_list(response.answer)
            return LLM.Response(answers=answers, input_tokens=0, output_tokens=0)

        except Exception as ex:
            print(f'Error in Together.ask_for_open_list "{response.answer}": {ex} ')
            return EMPTY_LIST

    async def ask_for_ranked_list(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.Response:
        return await self.ask_for_open_list(system_prompt, question, temperature)

    async def ask_for_list(
        self,
        choices: int,
        question: str,
        safe_answer: str,
        temperature: float | None,
    ) -> LLM.Response:
        result = await self.ask_for_ranked_list(
            RANKED_LIST_SYS_PROMPT, question, temperature
        )
        if len(result.answers) > choices:
            print(
                f'Ignoring extra {len(result.answers) - choices} choices in Together.llama: {",".join(result.answers)}'
            )
            result.answers = result.answers[:choices]
        return result

    async def choice_from_pair(
        self,
        question: str,
        temperature: float,
        max_iterations: int,
        system_prompt=None,
    ) -> LLM.Choice:
        result = await self.ask_generic_question(
            CHOICE_SYS_PROMPT, question, temperature, False
        )
        return LLM.Choice(
            answer=self.clean_reply(result.answer),
            probability=result.probability,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    async def ask_generic_question_with_retries(
        self, system_prompt, question, temperature, is_json, max_retries=10
    ):
        pass

    async def conversation(self, questions, temperature) -> LLM.Conversation:
        return LLM.Conversation()

    @staticmethod
    def order_models(models: list[str]):
        if SUPPORTED_MODEL_INTERNAL_NAME in models:
            return [SUPPORTED_MODEL_INTERNAL_NAME]
        return []

    # pylint: disable=unused-argument
    @staticmethod
    def display_name(model: str) -> str:
        return SUPPORTED_MODEL_DISPLAY_NAME

    @staticmethod
    def extract_logprobs(completion: Any) -> float | None:
        if (
            completion
            and completion.choices
            and len(completion.choices) > 0
            and completion.choices[0].logprobs
            and completion.choices[0].logprobs.token_logprobs
        ):
            return math.exp(sum(completion.choices[0].logprobs.token_logprobs))
        return None

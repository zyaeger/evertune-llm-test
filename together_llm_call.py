import math
import random
import together
from together import AsyncTogether
from dotenv import dotenv_values
from pydantic import BaseModel

from llm_call import *

SUPPORTED_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
SUPPORTED_MODEL_INTERNAL_NAME = 'llama-3.1-70B'
SUPPORTED_MODEL_DISPLAY_NAME = 'Llama-3.1'
RANKED_LIST_SYS_PROMPT = 'You are an assistant that returns a JSON object where the keys are the brand names and the values are the order you return them. Example output: {"foo": 1, "bar": 2}'
CHOICE_SYS_PROMPT = 'In one word answer the following question, strictly with the choice between two options'

class Choices(BaseModel):
    choices: dict[str, int]

NUMBERS_TO_REJECT = { str(i) for i in range(0, 20)}
EMPTY_LIST = LLM.Response(answers=[], input_tokens=0, output_tokens=0)
EMPTY_ANSWER = LLM.SimpleResponse(answer='', probability=None, input_tokens=0, output_tokens=0)


class Model(LLM):
    def __init__(self):
        self.__client = AsyncTogether(api_key=dotenv_values('.secrets')['TOGETHER_API_KEY'])

    @staticmethod
    def list_models() -> list[str]:
        return [SUPPORTED_MODEL_INTERNAL_NAME]

    @property
    def computed_model_name(self):
        return SUPPORTED_MODEL_INTERNAL_NAME

    @staticmethod
    def known_models() -> set[str]:
        return {SUPPORTED_MODEL_INTERNAL_NAME}

    def parallelism(self):
        return 100

    @property
    def has_logprob(self):
        return True

    async def ask_generic_question(self, system_prompt: str, question: str, temperature: float, is_json: bool) -> LLM.SimpleResponse:
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

                return LLM.SimpleResponse(
                    answer=response.choices[0].message.content,
                    probability=self.extract_logprobs(response),
                    input_tokens=0,  # TODO
                    output_tokens=0, # TODO
                )
            except together.error.TogetherException as ex:
                if ex.http_status in {429, 502, 503}:
                    seconds = int(pow(2, retry)*(1.0 + random.randint(0, 100)/100.0))
                    print(f"Llama on Together {ex.http_status}: waiting {seconds} seconds to avoid quota error attempt {retry}.")
                    await asyncio.sleep(seconds)
                else:
                    print(f"Error in Llama on Together: {ex}")
                    return EMPTY_ANSWER
            except Exception as ex:
                print(f"Error in Llama on Together: {ex}")
                return EMPTY_ANSWER

    async def ask_for_open_list(self, system_prompt: str, question: str, temperature: float) -> LLM.Response:
        response = await self.ask_generic_question(system_prompt, question, temperature, True)
        try:
            output  = json.loads(response.answer)
            choices = [(k, v) for k, v in output['choices'].items()]
            if ((not all(type(c[0]) is str for c in choices) or
                 not all(type(c[1]) is int for c in choices)) or
                    not all(0 <= c[1] <= len(choices) for c in choices) or
                    any(c[0] in NUMBERS_TO_REJECT for c in choices)):
                print(f'Ignoring answer from together_llama: {response.answer}')
                return EMPTY_LIST

            answers = [k for k, _ in sorted(choices, key=lambda k : k[1])]
            return LLM.Response(answers=answers, input_tokens=0, output_tokens=0)

        except Exception as ex:
            print(f'Error in Together.ask_for_open_list "{response.answer}": {ex} ')
            return EMPTY_LIST

    async def ask_for_ranked_list(self, system_prompt: str, question: str, temperature: float) -> LLM.Response:
        return await self.ask_for_open_list(RANKED_LIST_SYS_PROMPT, question, temperature)

    async def ask_for_list(self, choices: int, question: str, safe_answer: str, temperature: float | None) -> LLM.Response:
        result = await self.ask_for_ranked_list(RANKED_LIST_SYS_PROMPT, question, temperature)
        if len(result.answers) > choices:
            print(f'Ignoring extra {len(result.answers) - choices} choices in Together.llama: {",".join(result.answers)}')
            result.answers = result.answers[:choices]
        return result

    async def choice_from_pair(self, question: str, temperature: float, max_iterations: int, system_prompt = None) -> LLM.Choice:
        result = await self.ask_generic_question(CHOICE_SYS_PROMPT, question, temperature, False)
        return LLM.Choice(
            answer=self.clean_reply(result.answer),
            probability=result.probability,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    @staticmethod
    def order_models(models: list[str]):
        if SUPPORTED_MODEL_INTERNAL_NAME in models:
            return [SUPPORTED_MODEL_INTERNAL_NAME]
        return []

    @staticmethod
    def display_name(model: str) -> str:
        return SUPPORTED_MODEL_DISPLAY_NAME

    @staticmethod
    def extract_logprobs(completion) -> float | None:
        if (completion and completion.choices and len(completion.choices) > 0
                and completion.choices[0].logprobs and completion.choices[0].logprobs.token_logprobs):
            return math.exp(sum(completion.choices[0].logprobs.token_logprobs))
        return None

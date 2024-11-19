import asyncio
import json
from dataclasses import dataclass, field
from math import sqrt

class LLM:
    @dataclass
    class SimpleResponse:
        answer: str
        probability: float | None
        input_tokens: int
        output_tokens: int

    @dataclass
    class Response:
        answers: [str]
        input_tokens: int
        output_tokens: int

    @dataclass
    class Choice:
        answer: str
        probability: float
        input_tokens: int
        output_tokens: int


    @dataclass
    class Conversation:
        @dataclass
        class Answer:
            ordinal: int
            question: str
            answers: list[str]
            input_tokens: int
            output_tokens: int

        conversation: dict[int,Answer] = field(default_factory=dict[int,Answer])
        input_tokens: int = 0
        output_tokens: int = 0

        def add(self, answer: Answer):
            self.conversation[answer.ordinal] = answer
            self.input_tokens += answer.input_tokens
            self.output_tokens += answer.output_tokens

    @property
    def computed_model_name(self):
        raise NotImplementedError()

    async def ask_for_list(self, choices: int, question: str, safe_answer: str, temperature: float | None) -> Response:
        raise NotImplementedError()

    async def conversation(self, questions: list[str], temperature: float | None) -> Conversation:
        raise NotImplementedError()

    async def ask_generic_question(self, system_prompt: str, question: str, temperature: float, is_json: bool) -> SimpleResponse:
        raise NotImplementedError()

    async def ask_for_open_list(self, system_prompt: str, question: str, temperature: float) -> Response:
        result = await self.ask_generic_question(system_prompt, question, temperature, is_json=True)
        return self.Response(
            answers=self.parse_json_ranked_list(result.answer),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens
        )

    async def ask_for_ranked_list(self, system_prompt: str, question: str, temperature: float) -> Response:
        return await self.ask_for_open_list(system_prompt, question, temperature)

    @staticmethod
    def clean_reply(text: str) -> str:
        return text.strip(' ."1234567890\t\r\n*-:;•').strip("'")

    @staticmethod
    def parse_list(text: str) -> list[str]:
        splittered = text.split(',')
        if len(splittered) == 1:
            splittered = text.split('\n')
        return [LLM.clean_reply(s) for s in splittered if len(s) > 0]

    @staticmethod
    def known_models()  -> set[str]:
        return {"gpt-3.5-turbo", "gpt-4", "gemini-pro"}

    @staticmethod
    def report_models():
        return ["gpt-3.5-turbo", "gpt-4", "gemini-pro"]

    def parallelism(self):
        return 1

    @property
    def has_logprob(self):
        return True

    @staticmethod
    def parse_json_ranked_list(text: str):
        json_data = text.strip('` \n')
        if json_data.startswith('json'):
            json_data = json_data[4:]
        the_dict: dict[str, int] = json.loads(json_data)
        the_list = sorted([(n, v) for n, v in the_dict.items()], key=lambda k: k[1])
        answers = [a[0] for a in the_list]
        return answers

    @staticmethod
    def wald(p: float, n: int) -> float:
        try:
            match n:
                case 0: return 0.0
                case _ if p < 0 or p > 1: return 0.0
                case _: return 1.96*sqrt((p*(1-p))/n)
        except ValueError:
            return 0.0

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



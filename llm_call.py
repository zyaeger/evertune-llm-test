import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import sqrt


class LLM(ABC):
    @dataclass
    class SimpleResponse:
        answer: str
        probability: float | None
        input_tokens: int
        output_tokens: int

    @dataclass
    class Response:
        answers: list[str]
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

        conversation: dict[int, Answer] = field(
            default_factory=dict[int, Answer]
        )
        input_tokens: int = 0
        output_tokens: int = 0

        def add(self, answer: Answer) -> None:
            self.conversation[answer.ordinal] = answer
            self.input_tokens += answer.input_tokens
            self.output_tokens += answer.output_tokens

    @property
    def computed_model_name(self) -> str:
        pass

    @abstractmethod
    async def ask_for_list(
        self,
        choices: int,
        question: str,
        safe_answer: str,
        temperature: float | None,
    ) -> Response:
        pass

    @abstractmethod
    async def conversation(
        self, questions: list[str], temperature: float | None
    ) -> Conversation:
        pass

    @abstractmethod
    async def ask_generic_question(
        self,
        system_prompt: str,
        question: str,
        temperature: float,
        is_json: bool,
    ) -> SimpleResponse:
        pass

    @abstractmethod
    async def ask_for_open_list(
        self, system_prompt: str, question: str, temperature: float
    ) -> Response:
        pass

    @abstractmethod
    async def ask_for_ranked_list(
        self, system_prompt: str, question: str, temperature: float
    ) -> Response:
        pass

    @abstractmethod
    async def ask_generic_question_with_retries(
        self,
        system_prompt: str,
        question: str,
        temperature: float,
        is_json: bool,
        max_retries: int = 10,
    ) -> SimpleResponse:
        pass

    @abstractmethod
    async def choice_from_pair(
        self,
        question: str,
        temperature: float,
        max_iterations: int,
        system_prompt=None,
    ) -> Choice:
        pass

    @staticmethod
    def clean_reply(text: str) -> str:
        return text.strip(' ."1234567890\t\r\n*-:;•').strip("'")

    @staticmethod
    def parse_list(text: str) -> list[str]:
        splittered = text.split(",")
        if len(splittered) == 1:
            splittered = text.split("\n")
        return [LLM.clean_reply(s) for s in splittered if len(s) > 0]

    @staticmethod
    def known_models() -> set[str]:
        return {"gpt-3.5-turbo", "gpt-4", "gemini-pro"}

    @staticmethod
    def report_models() -> list[str]:
        return ["gpt-3.5-turbo", "gpt-4", "gemini-pro"]

    def parallelism(self) -> int:
        return 1

    @property
    def has_logprob(self) -> bool:
        return True

    @staticmethod
    def parse_json_ranked_list(text: str) -> list[str]:
        output = json.loads(text)
        choices = list(output["choices"].items())
        if (
            (
                not all(isinstance(c[0], str) for c in choices)
                or not all(isinstance(c[1], int) for c in choices)
            )
            or not all(0 <= c[1] <= len(choices) for c in choices)
            or any(c[0] in {str(i) for i in range(0, 20)} for c in choices)
        ):
            print(f"Ignoring answer from LLM: {text}")
            return []

        answers = [k for k, _ in sorted(choices, key=lambda k: k[1])]
        return answers

    @staticmethod
    def wald(p: float, n: int) -> float:
        try:
            match n:
                case 0:
                    return 0.0
                case _ if p < 0 or p > 1:
                    return 0.0
                case _:
                    return 1.96 * sqrt((p * (1 - p)) / n)
        except ValueError:
            return 0.0

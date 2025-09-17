import math

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from constants import (
    CHOICE_SYS_PROMPT,
    EMPTY_ANSWER,
    EMPTY_LIST,
    RANKED_LIST_SYS_PROMPT,
    Choices,
)
from llm_call import LLM


SUPPORTED_MODEL = "gemini-2.5-flash"

load_dotenv()


class Model(LLM):
    def __init__(self) -> None:
        self.__client = genai.Client(
            http_options=types.HttpOptions(
                api_version="v1",
                retry_options=types.HttpRetryOptions(attempts=10),
            )
        )

    @property
    def computed_model_name(self) -> str:
        return SUPPORTED_MODEL

    @property
    def parallelism(self) -> int:
        return 15000

    async def ask_for_list(
        self,
        choices: int,
        question: str,
        safe_answer: str,
        temperature: float | None,
    ) -> LLM.Response:
        # Ideally, some logic here to determine ranked vs open list
        # and call accordingly
        result = await self.ask_for_ranked_list(
            RANKED_LIST_SYS_PROMPT, question, temperature
        )
        if len(result.answers) > choices:
            print(
                f'Ignoring extra {len(result.answers) - choices} choices in Gemini: {",".join(result.answers)}'
            )
            result.answers = result.answers[:choices]
        return result

    async def conversation(
        self, questions: list[str], temperature: float | None
    ) -> LLM.Conversation:
        chat = self.__client.aio.chats.create(
            model=SUPPORTED_MODEL,
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )
        conversation = LLM.Conversation()
        try:
            for i, q in enumerate(questions):
                response = await chat.send_message(q)
                answer = conversation.Answer(
                    ordinal=i,
                    question=q,
                    answers=[part.text for part in response.parts],
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                )
                conversation.add(answer)
        except errors.APIError as exc:
            print("Error in Gemini chat:", exc)
            return LLM.Conversation()

        return conversation

    async def ask_generic_question(
        self,
        system_prompt: str,
        question: str,
        temperature: float,
        is_json: bool,
    ) -> LLM.SimpleResponse:
        logprobs = 1 if not is_json else 0
        try:
            if not is_json:
                response = await self.__client.aio.models.generate_content(
                    model=self.computed_model_name,
                    contents=question,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        response_logprobs=self.has_logprob,
                        logprobs=logprobs,
                        response_mime_type="text/plain",
                    ),
                )
            else:
                response = await self.__client.aio.models.generate_content(
                    model=self.computed_model_name,
                    contents=question,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        response_logprobs=self.has_logprob,
                        logprobs=logprobs,
                        response_mime_type="application/json",
                        response_json_schema=Choices.model_json_schema(),
                    ),
                )
            return LLM.SimpleResponse(
                answer=response.text,
                probability=math.exp(
                    response.candidates[0]
                    .logprobs_result.chosen_candidates[0]
                    .log_probability
                ),
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
            )
        except errors.APIError as exc:
            print(f"Error in Gemini: {exc}")
            return EMPTY_ANSWER

    async def ask_for_open_list(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.Response:
        # Not sure why call stack is so long
        # list -> ranked_list -> open_list -> generic_question
        # seems like overkill, I'd put some logic in ask_for_list
        # to determine ranked vs open list and call accordingly
        pass

    # pylint: disable=broad-exception-caught
    async def ask_for_ranked_list(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.Response:
        result = await self.ask_generic_question(
            system_prompt, question, temperature, is_json=True
        )
        try:
            answers = self.parse_json_ranked_list(result.answer)
            return self.Response(
                answers=answers,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        except Exception as ex:
            print("Error when parsing json response:", ex)
            return EMPTY_LIST

    async def choice_from_pair(
        self,
        question: str,
        temperature: float,
        max_iterations: int,
        system_prompt=None,
    ) -> LLM.Choice:
        if not system_prompt:
            system_prompt = CHOICE_SYS_PROMPT

        result = await self.ask_generic_question(
            system_prompt, question, temperature, False
        )
        return LLM.Choice(
            answer=self.clean_reply(result.answer),
            probability=result.probability,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    async def ask_generic_question_with_retries(
        self, system_prompt, question, temperature, is_json, max_retries=10
    ) -> LLM.SimpleResponse:
        # Gemini SDK supports retry_options in config
        # We can set retry limits at the client level
        pass

    @staticmethod
    def known_models() -> set[str]:
        return {SUPPORTED_MODEL}

    @staticmethod
    def report_models() -> list[str]:
        return [SUPPORTED_MODEL]

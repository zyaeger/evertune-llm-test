import asyncio

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from constants import RANKED_LIST_SYS_PROMPT, CHOICE_SYS_PROMPT, Choices
from llm_call import LLM


SUPPORTED_MODEL = "gemini-2.5-flash"

load_dotenv()

EMPTY_LIST = LLM.Response(answers=[], input_tokens=0, output_tokens=0)
EMPTY_ANSWER = LLM.SimpleResponse(
    answer="", probability=None, input_tokens=0, output_tokens=0
)


class Model(LLM):
    def __init__(self):
        self.__client = genai.Client(
            http_options=types.HttpOptions(
                api_version="v1",
                retry_options=types.HttpRetryOptions(attempts=10),
            )
        )

    @property
    def computed_model_name(self) -> str:
        return SUPPORTED_MODEL

    def parallelism(self) -> int:
        return 1000

    async def ask_for_list(
        self, choices: int, question: str, safe_answer: str, temperature: float | None
    ) -> LLM.Response:
        result = await self.ask_for_ranked_list(RANKED_LIST_SYS_PROMPT, question, temperature)
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
                response_logprobs=True,
                logprobs=1,
            )
        )
        conversation = LLM.Conversation()
        for idx, q in enumerate(questions):
            try:
                response = await chat.send_message(q)
                answer = conversation.Answer(
                    ordinal=idx,
                    question=q,
                    answers=[part.text for part in response.candidates[0].content.parts],
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                )
                conversation.add(answer)
            except errors.APIError as exc:
                print(f"Error in Gemini:", exc)
                continue
        
        return conversation


    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float, is_json: bool
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
                        response_logprobs=True,
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
                        response_logprobs=True,
                        logprobs=logprobs,
                        response_mime_type="application/json",
                        response_json_schema=Choices.model_json_schema(),
                    ),
                )
            return LLM.SimpleResponse(
                answer=response.text,
                probability=response.candidates[0].logprobs_result,
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
            )
        except errors.APIError as exc:
            print(f"Error in Gemini: {exc}")
            return EMPTY_ANSWER

    async def ask_for_open_list(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.Response:
        result = await self.ask_generic_question(
            system_prompt, question, temperature, is_json=True
        )
        return self.Response(
            answers=self.parse_json_ranked_list(result.answer),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    async def ask_for_ranked_list(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.Response:
        return await self.ask_for_open_list(system_prompt, question, temperature)

    # async def ask_generic_question_with_retries(
    #     self,
    #     system_prompt: str,
    #     question: str,
    #     temperature: float,
    #     is_json: bool,
    #     max_retries: int = 10,
    # ):
    #     for retry in range(0, max_retries):
    #         try:
    #             return await self.ask_generic_question(
    #                 system_prompt, question, temperature, is_json
    #             )
    #         except Exception as ex:
    #             if retry == max_retries - 1:
    #                 print(f"No more retries for {self.computed_model_name}: {ex}")
    #                 raise ex

    #             wait = pow(2, retry)
    #             print(
    #                 f"Waiting {wait} seconds for {self.computed_model_name} to avoid {ex}"
    #             )
    #             await asyncio.sleep(wait)

    async def choice_from_pair(
        self, question: str, temperature: float, max_iterations: int, system_prompt=None
    ):
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

    @staticmethod
    def known_models() -> set[str]:
        return {SUPPORTED_MODEL}

    @staticmethod
    def report_models() -> list[str]:
        return [SUPPORTED_MODEL]

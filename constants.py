from pydantic import BaseModel

from llm_call import LLM

RANKED_LIST_SYS_PROMPT = 'You are an assistant that returns a JSON object where the keys are the brand names and the values are the order you return them. Example output: {"foo": 1, "bar": 2}'
NEW_RANKED_LIST_SYS_PROMPT = 'You are a marketing assistant tasked with identifying top brands. Do not confuse brands with products. Return a JSON object where the keys are the brand names and the values are the order you return them. Example output: {"foo": 1, "bar": 2}'
CHOICE_SYS_PROMPT = "In one word answer the following question, strictly with the choice between two options. Must choose one."

EMPTY_LIST = LLM.Response(answers=[], input_tokens=0, output_tokens=0)
EMPTY_ANSWER = LLM.SimpleResponse(
    answer="", probability=None, input_tokens=0, output_tokens=0
)


class Choices(BaseModel):
    choices: dict[str, int]

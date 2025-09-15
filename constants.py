from pydantic import BaseModel

RANKED_LIST_SYS_PROMPT = 'You are an assistant that returns a JSON object where the keys are the brand names and the values are the order you return them. Example output: {"foo": 1, "bar": 2}'
CHOICE_SYS_PROMPT = "In one word answer the following question, strictly with the choice between two options"


class Choices(BaseModel):
    choices: dict[str, int]

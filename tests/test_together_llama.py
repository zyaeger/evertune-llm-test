import asyncio
import time

import pytest
from tabulate import tabulate

from together_llm_call import Model as TLlama


@pytest.mark.asyncio
async def test_choice():
    model = TLlama()
    answer = await model.choice_from_pair(
        "Which car is the best - Volvo or Saab?", 1.0, 10
    )
    print(answer)


@pytest.mark.asyncio
async def test_choices_at_scale():
    model = TLlama()
    iterations = model.parallelism() * 2
    start_time = time.perf_counter()
    counter = [0]
    stats = {"Bad": 0}

    async def run_calls(loop):
        while counter[0] < iterations:
            counter[0] += 1
            choice = await model.choice_from_pair(
                "In one word, which vintage car is the best - Volvo or Saab? Must choose one. Do not include any explanations",
                1.0,
                10,
            )
            answer = choice.answer
            if answer not in {"Volvo", "Saab"}:
                stats["Bad"] += 1
                counter[0] -= 1
            elif answer not in stats:
                stats[answer] = 1
            else:
                stats[answer] = stats[answer] + 1

    await asyncio.gather(*[run_calls(i) for i in range(0, model.parallelism())])
    results = sorted(
        [(k, v) for k, v in stats.items()], key=lambda k: k[1], reverse=True
    )
    elapsed_time = time.perf_counter() - start_time
    print(
        f"\nfinished {iterations} -> {len(results)} calls in {elapsed_time:.2f} seconds"
    )
    print("\n".join(f"{s}: {n}" for (s, n) in results))


@pytest.mark.asyncio
async def test_list_stats():
    iterations = 10
    questions = [
        "Which [insert written number] brands stand out to you the most in [insert product category]?",
        "When you hear [insert product category], which [insert written number] brands immediately come to your mind?",
        "Think of [insert product category]. What are the first [insert written number] brands that you think of?",
        "In your opinion, what are the [insert written number] most memorable brands in [insert product category]?",
    ]
    number = 5
    category = "Luxury SUVs"
    total_rounds = [iterations * len(questions)]
    model = TLlama()
    stats = {}

    async def run_calls(loop):
        while total_rounds[0] > 0:
            qq = (
                questions[total_rounds[0] % len(questions)]
                .replace("[insert written number]", str(number))
                .replace("[insert product category]", category)
            )
            total_rounds[0] -= 1
            answers = await model.ask_for_list(number, qq, "", 0.9)
            for i, answer in enumerate(answers.answers):
                assert i < number
                stats.setdefault(answer, [0 for _ in range(0, number)])
                stat = stats[answer]
                stat[i] += 1
                stats[answer] = stat

    start_time = time.perf_counter()
    await asyncio.gather(*[run_calls(i) for i in range(0, model.parallelism())])
    elapsed_time = time.perf_counter() - start_time
    print(
        f"\nfinished {iterations} -> {iterations*len(questions)} calls in {elapsed_time:.2f} seconds"
    )
    print(
        tabulate(
            [
                [s] + n
                for s, n in sorted(
                    [(s, n) for s, n in stats.items()], key=lambda k: k[1], reverse=True
                )
            ],
            headers=["Brand"] + [f"#{i + 1}" for i in range(0, number)],
            tablefmt="presto",
        )
    )

#!/usr/bin/env python3

# Synthesize annotated Dafny programs based on a seed set.

import argparse
import json
import random

import openai
from tqdm import tqdm

from annotator import load_benchmarks
from program import DafnyProgram, VerificationOutcome


client = openai.Client()


def extract_program(response: str) -> str:
    if '// BEGIN DAFNY' in response:
        response = response.split('// BEGIN DAFNY')[1]
    if '// END DAFNY' in response:
        response = response.split('// END DAFNY')[0]
    return response.strip()


def generate_dafny_program(
    model: str,
    seed_examples: list[str]
) -> str:
    # Construct the initial system message
    system_message = """You are a creative and correct Dafny program generator. Your task is to generate a new interesting example of a correct, annotated, complete Dafny program."""

    # Construct the user message with seed examples
    user_message = "Here are some example Dafny programs for inspiration:"

    for i, p in enumerate(seed_examples):
        user_message += f"\n### Program {i + 1}:\n// BEGIN DAFNY\n\n{p}\n\n// END DAFNY"


    user_message += "\n\nNow, generate an idea for a new Dafny file. Just describe it in natural language, don't implement any code yet. Prefer ideas that will take only a few functions at most (even just one is ok). Draw inspiration from the programs above: think about creative ways to extend or modify them, not proposing anything overly convoluted, but still different from the specific examples. Before writing your idea, describe how it will borrow, extend or modify ideas from the given programs."

    # First request: Generate a high-level idea
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        max_tokens=256,
    )

    idea = response.choices[0].message.content

    # Second request: Generate the complete Dafny program
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": idea},
            {"role": "user", "content": "Now, generate a complete, annotated Dafny program that implements the idea you just described. Before your program, provide a brief explanation of the specification of each function you will implement: what are its pre-conditions (requires clauses) and post-conditions (ensures clauses). Then, implement your plan.  Begin your program with a comment line // BEGIN DAFNY, and end it with // END DAFNY, just like the examples above. Your code should not have a main function or unit tests."}
        ],
        temperature=0,
        max_tokens=2048,
    )

    dafny_program = response.choices[0].message.content

    return idea, dafny_program


def generate_synthetic_dataset(
    model: str,
    seed_dataset: list[str],
    max_seeds: int,
    num_programs: int,
    suffix: str,
) -> list[dict]:
    synthetic_dataset = []
    good_programs = 0

    OUTPUT_PATH = f"{model}-{num_programs}{"-" if suffix else ""}{suffix}.json"

    for i in tqdm(range(num_programs)):
        random_seed = random.sample(seed_dataset,
                                    random.randint(2, max_seeds))

        idea, response = generate_dafny_program(model, random_seed)
        dafny_program = extract_program(response)

        p = DafnyProgram(dafny_program)
        outcome = p.verify()
        print('Idea:', idea)
        print('Program:', dafny_program)
        print('Outcome:', outcome)

        good_programs += outcome == VerificationOutcome.SUCCESS

        synthetic_dataset.append({
            "idea": idea,
            "seed_examples": random_seed,
            "response": response,
            "program": dafny_program,
            "dafny_feedback": str(outcome)
        })

        with open(OUTPUT_PATH, "w") as f:
            json.dump(synthetic_dataset, f, indent=4)

    print('Generated', num_programs, 'with', good_programs, 'that Dafny verified.')


def main():
    parser = argparse.ArgumentParser(description="Synthesize annotated Dafny programs based on a seed set.")
    parser.add_argument("--model", type=str, help="LLM to use", default='gpt-4o-2024-08-06')
    parser.add_argument("--N", type=int, default=100, help="The number of synthetic programs to generate.")
    parser.add_argument("--max-seeds", type=int, default=2, help="Max number of seed examples given in context.")
    parser.add_argument("--suffix", type=str, default='', help="Suffix for the output file.")

    args = parser.parse_args()

    benchmark = load_benchmarks('./DafnyBench/programs')
    seed_dataset = [str(b) for b in benchmark]

    generate_synthetic_dataset(args.model, seed_dataset, max_seeds=args.max_seeds, num_programs=args.N,
                               suffix=args.suffix)


if __name__ == "__main__":
    main()

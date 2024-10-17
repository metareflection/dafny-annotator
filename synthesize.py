#!/usr/bin/env python3

# Synthesize annotated Dafny programs based on a seed set.

import argparse
import json
import random
import os
from pathlib import Path

import openai
from tqdm import tqdm

from annotator import load_benchmarks
from program import DafnyProgram, VerificationOutcome
import training


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


def extract_programs_from_graph(input_path: str,
                                output_dir: str,
                                include_unproven: bool = True):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                json_file_name = os.path.splitext(os.path.basename(json_path))[0]

                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    if not isinstance(data, list):
                        print(f"Warning: {json_path} does not contain a list. Skipping.")
                        continue

                    for index, obj in enumerate(data):
                        if 'program' in obj and (
                                'SUCCESS' in obj['dafny_feedback'] or
                                (include_unproven and 'GOAL_UNPROVEN' in obj['dafny_feedback'])):
                            output_file = f"{json_file_name}-{index}.dfy"
                            output_path = os.path.join(output_dir, output_file)

                            with open(output_path, 'w') as f:
                                f.write(obj['program'])

                            print(f"Wrote {output_path}")
                            processed_count += 1

                except Exception as e:
                    print(f"Error processing {json_path}: {str(e)}")

    print(processed_count, 'Dafny files written to', output_dir)

    training.make_direct_examples(output_dir, out_path / 'direct_finetuning_examples.json')


def main():
    parser = argparse.ArgumentParser(description="Synthesize annotated Dafny programs based on a seed set.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic Dafny programs")
    generate_parser.add_argument("--model", type=str, help="LLM to use", default='gpt-4o-2024-08-06')
    generate_parser.add_argument("--N", type=int, default=100, help="The number of synthetic programs to generate.")
    generate_parser.add_argument("--max-seeds", type=int, default=2, help="Max number of seed examples given in context.")
    generate_parser.add_argument("--suffix", type=str, default='', help="Suffix for the output file.")

    extract_parser = subparsers.add_parser("extract", help="Extract programs from JSON files")
    extract_parser.add_argument("input_path", type=str, help="Path to the input JSON files")
    extract_parser.add_argument("--output-dir", type=str, default="DafnySynth",
                                help="Directory to dump programs extracted from the graph.")
    extract_parser.add_argument("--include-unproven", action="store_true",
                                help="Include programs that don't fully verify.")

    args = parser.parse_args()

    if args.command == "generate":
        benchmark = load_benchmarks('./DafnyBench/programs')
        seed_dataset = [str(b) for b in benchmark]
        generate_synthetic_dataset(args.model, seed_dataset, max_seeds=args.max_seeds, num_programs=args.N,
                                   suffix=args.suffix)
    elif args.command == "extract":
        processed_count = extract_programs_from_graph(args.input_path, args.output_dir, args.include_unproven)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

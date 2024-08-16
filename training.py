#!/usr/bin/env python3

import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from torchtune.data import InstructTemplate
from torchtune.datasets import InstructDataset
from tqdm import tqdm
import openai

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from annotator import load_nontrivial_benchmarks, load_benchmarks
from program import VerificationOutcome
import completion


OPENAI = openai.Client()


class DafnyAnnotationTemplate(InstructTemplate):
    @classmethod
    def format(cls, sample: dict, column_map=None):
        return completion.make_prompt(sample['program'], sample.get('rationale'),
                                      [sample['output']])


def print_nontrivial(args):
    programs = load_nontrivial_benchmarks('DafnyBench/programs')
    print(len(programs), 'non-trivial programs')


def rationalize(program, annotation):
    prompt = [
        {"role": "system",
         "content": "You are a Dafny expert. The user will give you a sequence of Dafny program that is missing an annotation (assertion or invariant). They will also give you the missing assertion or invariant. Your job is to provide a very short, concise rationale that would be a useful hint for someone coming up with that assertion or invariant."},
        {
            "role": "user",
            "content": f"""Program: method intersperse(numbers: seq<int>, delimiter: int) returns (interspersed: seq<int>)
    ensures |interspersed| == if |numbers| > 0 then 2 * |numbers| - 1 else 0
    ensures forall i :: 0 <= i < |interspersed| ==> i % 2 == 0 ==>
                interspersed[i] == numbers[i / 2]
    ensures forall i :: 0 <= i < |interspersed| ==> i % 2 == 1 ==>
                interspersed[i] == delimiter
{{
    interspersed := [];
    for i := 0 to |numbers|
    {{
        if i > 0 {{
            interspersed := interspersed + [delimiter];
        }}
        interspersed := interspersed + [numbers[i]];
    }}
}}

Missing annotation: invariant |interspersed| == if i > 0 then 2 * i - 1 else 0"""
        },
        {
            "role": "assistant",
            "content": "Rationale: Adapt the first ensures clause as an invariant"
        },
        {
            "role": "user",
            "content": f"Program: {program}\nMissing annotation: {annotation}"
        }
    ]

    completion = OPENAI.chat.completions.create(
        model="gpt-4o",
        messages=prompt,
        max_tokens=100
    )

    return completion.choices[0].message.content


def make_rationalization_examples(args):
    programs = load_nontrivial_benchmarks('DafnyBench/programs')

    examples = []

    for p in tqdm(programs):
        examples.extend(p.extract_examples())

    finetuning_examples = []

    for p, a in examples:
        r = rationalize(p, a)

        finetuning_examples.append({
            'program': p.to_json_obj(),
            'completion': a,
            'rationale': r
        })

    with open('finetuning_examples.json', 'w') as f:
        json.dump(finetuning_examples, f, indent=2)


def make_direct_examples(args):
    programs = load_benchmarks('DafnyBench/programs')[args.skip:]
    examples = []

    for p in tqdm(programs):
        examples.extend(p.extract_examples())

    finetuning_examples = []

    for p, a in examples:
        # For now skip very long programs to avoid OOMs.
        if len(str(p)) > 1024:
            continue
        finetuning_examples.append({
            'program': str(p),
            'output': a,
            'rationale': None,
        })

    print('Extracted', len(finetuning_examples), 'training examples')
    with open('direct_finetuning_examples.json', 'w') as f:
        json.dump(finetuning_examples, f, indent=2)


def merge_model(args):
    model = AutoModelForCausalLM.from_pretrained('./llama3-8b', device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained('./llama3-8b', device_map="auto").eval()

    _ = model.load_adapter("./llama3-8b/adapter_0.pt", adapter_name="ft")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLM name or path')
    parser.add_argument('--rationalize', action='store_true', help='Generate rationalization examples')
    parser.add_argument('--extract-direct', action='store_true', help='Generate rationalization examples')
    parser.add_argument('--nontrivial', action='store_true', help='Filter and save non-trivial benchmarks')
    parser.add_argument('--skip', type=int, default=200,
                        help='How many programs to skip (e.g. avoid test set)')
    parser.add_argument('--merge', action='store_true', help='Merge LoRA adapter with original model')

    args = parser.parse_args()

    if args.rationalize:
        make_rationalization_examples(args)
    elif args.extract_direct:
        make_direct_examples(args)
    elif args.nontrivial:
        print_nontrivial(args)
    elif args.merge:
        merge_model(args)

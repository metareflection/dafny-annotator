#!/usr/bin/env python3

import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import openai

from peft import PeftConfig, PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
import torch

from annotator import load_nontrivial_benchmarks, load_benchmarks
from program import VerificationOutcome
import completion


OPENAI = openai.Client()


def print_nontrivial(args):
    programs = load_nontrivial_benchmarks('DafnyBench/programs')
    print(len(programs), 'non-trivial programs')


def trim_program(program: str, max_length: int = 1000):
    if len(program) > max_length:
        return '...' + program[-max_length:]
    return program


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
    programs = load_benchmarks('DafnyBench/programs')[args.skip:]
    examples = []

    for p in tqdm(programs):
        examples.extend(p.extract_examples())

    finetuning_examples = []
    for p, a in examples:
        # For now skip long programs to avoid OOMs.
        if len(str(p)) > 2048:
            continue

        r = rationalize(p, a)

        finetuning_examples.append({
            'program': str(p),
            'output': a,
            'rationale': r
        })
    print('Extracted', len(finetuning_examples), 'training examples')

    with open('rationalized_finetuning_examples.json', 'w') as f:
        json.dump(finetuning_examples, f, indent=2)


def make_direct_examples(programs_path: str, output: str, skip: int = 0, max_length: int = 2048):
    programs = load_benchmarks(programs_path or 'DafnyBench/programs')[skip:]
    examples = []

    for p in tqdm(programs):
        examples.extend(p.extract_examples())

    finetuning_examples = []

    for p, a in examples:
        # For now skip very long programs to avoid OOMs.
        finetuning_examples.append({
            'program': trim_program(str(p), max_length),
            'output': a,
            'rationale': None,
        })

    output_path = output or 'direct_finetuning_examples.json'

    with open(output_path, 'w') as f:
        json.dump(finetuning_examples, f, indent=2)

    print('Extracted', len(finetuning_examples), 'training examples, saved to', output_path)


def merge_model(args):
    model = AutoModelForCausalLM.from_pretrained('./llama3-8b', device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained('./llama3-8b', device_map="auto").eval()

    _ = model.load_adapter("./llama3-8b/adapter_0.pt", adapter_name="ft")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def finetune(args):
    dataset = []

    for t in args.training_set:
        with open(t) as d_in:
            dataset.extend(json.load(d_in))

    print('Training set size:', len(dataset))

    for r in dataset:
        if args.with_rationales:
            rationale = r['rationale']
            # FIXME: this should be done during data generation.
            if rationale.startswith('Rationale: '):
                rationale = rationale[len('Rationale: '):]
            r['text'] = completion.make_prompt(r['program'], with_rationale=True) + ' [' + rationale + '] ' + r['output'] + '\n' + completion.END
        else:
            r['text'] = completion.make_prompt(r['program']) + ' ' + r['output'] + '\n' + completion.END

    dataset = Dataset.from_list(dataset)

    response_template = "\nAction:" # completion.RESPONSE_PREFIX

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForCompletionOnlyLM(
            response_template=tokenizer.encode(response_template)[2:],
            tokenizer=tokenizer)
    suffix = '-r' if args.with_rationales else ''
    output_dir = args.output

    sft_config = SFTConfig(
            dataset_text_field="text",
            max_seq_length=1024,
            output_dir=output_dir,
            num_train_epochs=3,
            )

    model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            )

    trainer = SFTTrainer(
            model,
            train_dataset=dataset,
            args=sft_config,
            peft_config=peft_config,
            data_collator=collator,
            )

    trainer.train()
    trainer.model.save_pretrained(output_dir + '-spt')
    merged_dir = output_dir + '-merged'
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)


def merge(input_files, output_file):
    combined_data = []

    for file_path in input_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            assert isinstance(data, list)
            combined_data.extend(data)

    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    print('Wrote', output_file, 'with', len(combined_data), 'examples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLM name or path', default='meta-llama/Meta-Llama-3.1-8B')
    parser.add_argument('--merge-data', action='store_true', help='Merge training sets')
    parser.add_argument('--rationalize', action='store_true', help='Generate rationalization examples')
    parser.add_argument('--extract-direct', action='store_true', help='Generate direct prediction examples')
    parser.add_argument('--nontrivial', action='store_true', help='Filter and save non-trivial benchmarks')
    parser.add_argument('--skip', type=int, default=200,
                        help='How many programs to skip (e.g. avoid test set)')
    parser.add_argument('--merge', action='store_true', help='Merge LoRA adapter with original model')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--programs', type=str,
                        help='Path to root directory containing Dafny programs to use for training.',
                        default='DafnyBench/programs')
    parser.add_argument('--finetune', action='store_true', help='Fine tune model on dataset of existing examples')
    parser.add_argument('--with-rationales', action='store_true', help='Fine tune model on dataset with rationales')
    parser.add_argument('--benchmark-dir', type=str, default='./DafnyBench/programs')
    parser.add_argument('--training-set', nargs='*', type=str)

    args = parser.parse_args()

    if args.rationalize:
        make_rationalization_examples(args)
    elif args.extract_direct:
        make_direct_examples(args.programs, args.output, args.skip)
    elif args.nontrivial:
        print_nontrivial(args)
    elif args.merge:
        merge_model(args)
    elif args.finetune:
        finetune(args)
    elif args.merge_data:
        merge(args.training_set, args.output)

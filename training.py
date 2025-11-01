#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

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

BATCH_SIZE=int(os.environ.get('BATCH_SIZE', '4'))
VFP_PROMPT = os.environ.get('VFP_PROMPT', 'false') != 'false'
MULTIGPU = os.environ.get('MULTIGPU', 'false') != 'false'


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
         "content": f"You are a Dafny expert. The user will give you a sequence of Dafny program that is missing an annotation (assertion or invariant{' or helper lemma call' if VFP_PROMPT else ''}). They will also give you the missing assertion or invariant{' or helper lemma call' if VFP_PROMPT else ''}. Your job is to provide a very short, concise rationale that would be a useful hint for someone coming up with that assertion or invariant{' or helper lemma call' if VFP_PROMPT else ''}."},
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

    OPENAI = openai.Client()

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


def make_direct_examples(
        programs_path: str,
        output_path: str,
        skip: int = 0,
        max_length: int = 2048,
        unique: bool = False,
        localized: bool = False):
    seen_annotations = set()

    programs = load_benchmarks(programs_path or 'DafnyBench/programs')[skip:]
    examples = []

    for p in tqdm(programs):
        examples.extend(p.extract_examples(localized=localized))

    finetuning_examples = []

    for p, a in examples:
        a = a.strip()
        if unique and a in seen_annotations:
            continue
        seen_annotations.add(a)
        finetuning_examples.append({
            'program': trim_program(str(p), max_length),
            'output': a,
            'rationale': None,
        })

    output_path = output_path or 'direct_finetuning_examples.json'

    with open(output_path, 'w') as f:
        json.dump(finetuning_examples, f, indent=2)

    print('Extracted', len(finetuning_examples), 'training examples, saved to', output_path)


def extract_programs_from_graph(json_path: str,
                                output_dir: str,
                                include_unproven: bool = True):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    json_file_name = os.path.splitext(os.path.basename(json_path))[0]

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict) or not data.get('nodes'):
            print(f"Skipping {json_path} (not a serialized edit graph)")
            return

        data = data['nodes']

        for index, obj in enumerate(data):
            if obj['type'] == 'program':
                output_file = f"{json_file_name}-{obj['id']}.dfy"
                output_path = os.path.join(output_dir, output_file)

                with open(output_path, 'w') as f:
                    f.write(obj['content'])

                processed_count += 1

    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")

    print(processed_count, 'Dafny files written to', output_dir)


def extract_examples_from_graph(graph_path: str, output: str, localized: bool = False):
    output_dir = os.path.dirname(output)
    print('Output:', output)
    extract_path = output_dir + '-programs'
    os.makedirs(extract_path, exist_ok=True)
    extract_programs_from_graph(graph_path, extract_path)
    make_direct_examples(extract_path, output, unique=True, localized=localized)


peft_config = LoraConfig(
    r=128,
    use_rslora=True,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_config_qwen_coder = LoraConfig(
    r=128,
    use_rslora=True,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ])

peft_config_gemma = LoraConfig(
    r=128,
    use_rslora=True,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "down_proj",
        "o_proj",
        "k_proj",
        "q_proj",
        "gate_proj",
        "up_proj",
        "v_proj",
    ],
)

peft_config_deepseek = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
        #"lm_head"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

def finetune(args):
    dataset = []

    for file_path in args.training_set:
        with open(file_path) as d_in:
            dataset.extend(json.load(d_in))
    print('Training set size:', len(dataset))

    for r in dataset:
        if args.with_rationales:
            rationale = r['rationale']
            # FIXME: this should be done during data generation.
            if rationale.startswith('Rationale: '):
                rationale = rationale[len('Rationale: '):]
            r['text'] = completion.make_prompt(r['program'], with_rationale=True, localized=args.localized) + ' [' + rationale + '] ' + r['output'] + '\n' + completion.END
        else:
            r['text'] = completion.make_prompt(r['program'], localized=args.localized) + ' ' + r['output'] + '\n' + completion.END

    dataset = Dataset.from_list(dataset)

    response_template = "\nAction:"  # completion.RESPONSE_PREFIX

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForCompletionOnlyLM(
            response_template=tokenizer.encode(response_template)[2:],
            tokenizer=tokenizer)
    output_dir = args.output

    sft_config = SFTConfig(
            dataset_text_field="text",
            max_seq_length=1024,
            output_dir=output_dir + '-peft',
            num_train_epochs=3,
            bf16=True if MULTIGPU else False,
            per_device_train_batch_size=BATCH_SIZE
            )

    model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto" if not MULTIGPU else None,#'cuda',
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            )

    #if MULTIGPU:
    #    model.to('cuda')

    model_lower = args.model.lower()

    trainer = SFTTrainer(
            model,
            train_dataset=dataset,
            args=sft_config,
            peft_config=peft_config_gemma if 'gemma' in model_lower else peft_config_deepseek if 'deepseek' in model_lower else peft_config_qwen_coder if 'qwen' in model_lower and 'coder' in model_lower else peft_config,
            data_collator=collator,
            )

    trainer.train()
    trainer.model.save_pretrained(output_dir + '-peft')
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


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
    parser.add_argument('--localized', action='store_true', help='Localize annotations')
    parser.add_argument('--extract-direct', action='store_true', help='Generate direct prediction examples')
    parser.add_argument('--extract-direct-from-graph', action='store_true', help='Generate direct prediction examples from a synthetic graph')
    parser.add_argument('--nontrivial', action='store_true', help='Filter and save non-trivial benchmarks')
    parser.add_argument('--skip', type=int, default=200,
                        help='How many programs to skip (e.g. avoid test set)')
    parser.add_argument('--graph', type=str, help='Path to input synthetic graph')
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
        make_direct_examples(args.programs, args.output, args.skip, args.localized)
    elif args.extract_direct_from_graph:
        extract_examples_from_graph(args.graph, args.output, args.localized)
    elif args.nontrivial:
        print_nontrivial(args)
    elif args.finetune:
        finetune(args)
    elif args.merge_data:
        merge(args.training_set, args.output)

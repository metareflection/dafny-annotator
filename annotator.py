#!/usr/bin/env python3

import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from synchromesh import HuggingFaceModel, predict_constrained

from program import DafnyProgram, VerificationOutcome
from completion import DafnyActionCompletionEngine, make_prompt
from test_example import program, verification_prompt


def propose(model: HuggingFaceModel, program: DafnyProgram, num_samples: int) -> list[DafnyProgram]:
    comp_engine = DafnyActionCompletionEngine(str(program))
    unique_predictions = set()

    for _ in range(num_samples):
        prediction = predict_constrained(comp_engine, model, 1, True, stop_tokens=["\n"]).strip()
        unique_predictions.add(prediction)

    new_programs = []
    start_line = program.first_line()
    end_line = program.last_line()

    for prediction in unique_predictions:
        for line in range(start_line, end_line):
            new_program = program.insert(line, prediction)
            new_programs.append(new_program)

    return new_programs


def load_lm_for_verification(model_name: str) -> HuggingFaceModel:
    lm = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return HuggingFaceModel(lm, tokenizer=tokenizer, prompt_template='', temperature=1)


def load_benchmarks(path: str) -> list[DafnyProgram]:
    benchmarks = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.dfy'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    program_string = f.read()
                    program = DafnyProgram(program_string, file)
                    benchmarks.append(program)

    return benchmarks


def test():
    from cmdline import args

    benchmarks = load_benchmarks('DafnyBench/programs')
    print('Loaded', len(benchmarks), 'benchmarks')
    model = load_lm_for_verification(args.model)
    print('Loaded model', args.model)

    print('Dafny output on original benchmark: ', benchmarks[0].verify())
    print('Dafny output after stripping annotations: ', benchmarks[0].strip_annotations().verify())

    for i, gold_program in enumerate(benchmarks):
        program = gold_program.strip_annotations()
        try:
            model.prompt_template = make_prompt(program)
            new_programs = propose(model, program, 3)
            print(f'Proposed {len(new_programs)} programs for program #{i} - {program.name}')

            for j, new_program in enumerate(new_programs):
                outcome = new_program.verify()
                print(f'    {j}', outcome)

                if outcome != VerificationOutcome.FAIL:
                    print('Progress!')
                    print(new_program)
        except:
            print(f'Failed to propose for {program}, probably too long.')


if __name__ == '__main__':
    test()

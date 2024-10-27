#!/usr/bin/env python3

from tqdm import tqdm
import os
import json
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from synchromesh import HuggingFaceModel, predict_constrained

from program import DafnyProgram, VerificationOutcome
from completion import DafnyActionCompletionEngine, make_prompt, END
from test_example import program, verification_prompt


def propose(model: HuggingFaceModel, program: DafnyProgram, num_samples: int) -> list[DafnyProgram]:
    comp_engine = DafnyActionCompletionEngine(str(program))
    unique_predictions = set()

    for _ in range(num_samples):
        prediction = predict_constrained(comp_engine, model, 1, True, stop_tokens=["\n"]).strip()

        lines = [l.strip() for l in prediction.split('\n')]
        if not lines or lines[-1] != END:
            breakpoint()
        assert lines[-1] == END
        lines = lines[:-1]
        # Transform '[comment] <annotation>' into '<annotation> /* comment */'
        for i in range(len(lines)):
            if not lines[i].startswith('['):
                breakpoint()
            assert lines[i].startswith('[')
            comment, annotation = lines[i][1:].split(']', 1)
            lines[i] = f'{annotation} // {comment}'.strip()

        unique_predictions.update(lines)

    new_programs = []
    start_line = program.first_line()
    end_line = program.last_line()

    if start_line is None or end_line is None:
        return unique_predictions, new_programs

    for prediction in unique_predictions:
        for line in range(start_line, end_line):
            new_program = program.insert(line, prediction)
            new_programs.append(new_program)

    return unique_predictions, new_programs


def load_lm_for_verification(model_name: str) -> HuggingFaceModel:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)
    return HuggingFaceModel(lm, tokenizer=tokenizer, prompt_template='', temperature=1)


def load_benchmarks(
    path: str,
    seed: str = 'dafny-annotator'
) -> list[DafnyProgram]:
    benchmarks = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.dfy'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    program_string = f.read()
                    program = DafnyProgram(program_string, file)
                    benchmarks.append(program)

    random.seed(seed)
    random.shuffle(benchmarks)
    return benchmarks


def load_nontrivial_benchmarks(path: str) -> list[DafnyProgram]:
    if os.path.exists(os.path.join(path, 'nontrivial.json')):
        with open(os.path.join(path, 'nontrivial.json'), 'r') as f:
            nontrivial = json.load(f)
            return [DafnyProgram.from_json_obj(p) for p in nontrivial]
    b = load_benchmarks(path)
    print('Loaded', len(b), 'benchmarks. Filtering non-trivial ones')
    b = [p for p in tqdm(b) if p.strip_annotations().verify() != VerificationOutcome.SUCCESS]
    with open(os.path.join(path, 'nontrivial.json'), 'w') as f:
        json.dump([p.to_json_obj() for p in b], f)
    print('Filtered', len(b), 'non-trivial benchmarks')
    return b


def annotate(program: DafnyProgram,
             model: HuggingFaceModel,
             max_attempts: int = 10) -> DafnyProgram:

    annotated_program = program

    for _ in range(max_attempts):
        model.prompt_template = make_prompt(annotated_program)
        annotations, new_programs = propose(model, annotated_program, 1)
        added_annotation = False

        print('Trying', len(new_programs), 'proposals')
        for new_program in new_programs:
            feedback = new_program.verify()

            if feedback == VerificationOutcome.SUCCESS:
                return new_program

            if feedback == VerificationOutcome.GOAL_UNPROVEN:
                added_annotation = True
                annotated_program = new_program
                print('Progress:')
                print(annotated_program)
                break

        if not added_annotation:
            print('All proposals failed')

    return annotated_program


def test():
    from cmdline import args

    out = open('run.log', 'w')

    def log(line):
        out.write(line + '\n')
        out.flush()
        print(line)

    benchmarks = load_benchmarks('DafnyBench/programs')
    print('Loaded', len(benchmarks), 'benchmarks')
    model = load_lm_for_verification(args.model)

    log(f'*Model*: {args.model}')

    for i, gold_program in enumerate(benchmarks):
        try:
            log(f'# {gold_program.name}')

            program = gold_program.strip_annotations()

            dafny_output_with_annotations = gold_program.verify()
            dafny_output_stripped = program.verify()

            log(f'```dafny\n{gold_program.format_method_lines()}\n```')

            log(f'*Original Dafny result*: {dafny_output_with_annotations}')
            log(f'*Stripped Dafny result*: {dafny_output_stripped}')

            if dafny_output_stripped == VerificationOutcome.SUCCESS:
                log('Skipping already verified program')
                continue

            annotated_program = annotate(program, model)

            log('*Annotated program:*:')
            log(str(annotated_program))

            feedback = annotated_program.verify()
            log(f'*Feedback*: {feedback}')
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f'*Error*: {e}')
            import traceback; traceback.print_exc()
            print(f'Failed to propose for {program}, probably too long.')


if __name__ == '__main__':
    test()

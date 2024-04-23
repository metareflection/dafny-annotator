#!/usr/bin/env python3

import os

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

            model.prompt_template = make_prompt(program)
            annotations, new_programs = propose(model, program, 1)
            good_predictions = set()

            print(f'Proposed {len(annotations)} annotations, {len(new_programs)} programs for program #{i} - {program.name}')

            log('*Proposed annotations*:')
            for i, p in enumerate(annotations):
                log(f'1. {p}')

            overall_outcome = 'fail :x:'
            children = 0

            for j, new_program in enumerate(new_programs):
                outcome = new_program.verify()
                print(f'    {j}', outcome)

                if outcome != VerificationOutcome.FAIL:
                    print('Progress!')
                    print(new_program)
                    children += 1
                    prediction = '; '.join(new_program.diff_annotations(program))

                    if outcome == VerificationOutcome.SUCCESS:
                        overall_outcome = 'success :white_check_mark:'
                        children = 0
                        good_predictions = {prediction}
                        break
                    else:
                        good_predictions.add(prediction)
                        overall_outcome = 'progress :arrow_right:'

            log(f'*Result*: {overall_outcome}')
            if children:
                log(f'*Children*: {children}')
            if good_predictions:
                log('*Verified predictions:*')
                for i, p in enumerate(good_predictions):
                    log(f'1. {p}')
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f'*Error*: {e}')
            breakpoint()
            import traceback; traceback.print_exc()
            print(f'Failed to propose for {program}, probably too long.')


if __name__ == '__main__':
    test()

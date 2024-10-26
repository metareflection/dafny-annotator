#!/usr/bin/env python3
"""Run evaluation pipelines."""

import os
import subprocess


def run(args: list[str]):
    """Run the given command and check that it succeeds."""
    subprocess.run(args, shell=True, check=True)


def print_done(result_path: str):
    """Print that the experiment is done."""
    print(f'✅ {result_path}')


def run_base_model_experiment(
    n_eval_programs: int,
    base_model: str
):
    """Run an experiment with a base model (no fine-tuning)."""
    model_name = base_model.split('/')[-1]
    result_path = os.path.join(f'results/base-{model_name}.json')

    if not os.path.exists(result_path):
        run(['python', 'search.py',
             '--num-programs', str(n_eval_programs),
             '--output', result_path,
             '--model', base_model])

    print_done(result_path)


def run_dafnybench_finetuning_experiment(
    n_eval_programs: int,
    base_model: str,
    finetuning_fraction: float,
):
    DAFNYBENCH_SIZE = 1326  # TODO: get this from the dataset.
    available_training_set = DAFNYBENCH_SIZE - n_eval_programs
    used_training_set = int(finetuning_fraction * available_training_set)
    skipped_training_set = available_training_set - used_training_set
    n_skip = skipped_training_set + n_eval_programs

    model_name = base_model.split('/')[-1]
    ft_percent = int(100 * finetuning_fraction)
    result_path = os.path.join(
        f'results/finetuned-{model_name}-db{ft_percent}.json')  # noqa

    training_set_path = f'data/finetuning_examples_{finetuning_fraction}.json'
    model_path = f'models/finetuned_{model_name}_db{ft_percent}'

    if not os.path.exists(result_path):
        # 1- Collect training set
        run(['python', 'training.py',
             '--extract-direct',
             '--skip', str(n_skip),
             '--output', training_set_path,
             ])

        # 2- Fine-tune
        run(['python', 'training.py',
             '--finetune',
             '--model', base_model,
             '--training-data', training_set_path,
             '--output', model_path,
             ])

        # 3- Evaluate
        run(['python', 'search.py',
             '--num-programs', str(n_eval_programs),
             '--output', result_path,
             '--model', model_path,
             ])

    print_done(result_path)


def main():
    N_EVAL_PROGRAMS = 326
    BASE_MODELS = [
        'meta-llama/Meta-Llama-3.1-8B',
    ]

    for m in BASE_MODELS:
        run_base_model_experiment(N_EVAL_PROGRAMS, m)

    EVAL_FRACTIONS = [.25, .5, 1.0]

    for m in BASE_MODELS:
        for f in EVAL_FRACTIONS:
            run_dafnybench_finetuning_experiment(N_EVAL_PROGRAMS, m, f)


    # TODO: Try with merging with DafnySynth

if __name__ == '__main__':
    main()

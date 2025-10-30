#!/usr/bin/env python3
"""Run evaluation pipelines."""

import os
import subprocess
from typing import Optional

RESULTS_DIR = os.environ.get("RESULTS_DIR", 'results')
MODELS_DIR = os.environ.get("MODELS_DIR", 'models')

VFP_SKETCH = os.environ.get('VFP_SKETCH', 'false') != 'false'
VFP_PROMPT = os.environ.get('VFP_PROMPT', 'false') != 'false'
VFP_MODULAR = os.environ.get('VFP_MODULAR', 'false') != 'false'
LOCALIZED = os.environ.get("LOCALIZED", 'false') != 'false'
MAYBE_LOCALIZED = ['--localized'] if LOCALIZED else []
MAYBE_SKETCH = ['--sketch'] if VFP_SKETCH else []

def run(args: list[str], check: bool = True):
    """Run the given command and check that it succeeds."""
    subprocess.run(args, check=check)


def print_done(result_path: str):
    """Print that the experiment is done."""
    print(f'✅ {result_path}')


def kill_dafny():
    """Kills all running Dafny processes.

    execute.py seems to leave some of these processes still running
    even after search.py finishes, and they don't let CUDA memory be
    freed up. This is a workaround, though we might want to fix execute.py
    """
    run(['killall', '-9', 'dafny'], check=False)


def run_base_model_experiment(
    n_eval_programs: int,
    base_model: str
):
    """Run an experiment with a base model (no fine-tuning)."""
    model_name = base_model.split('/')[-1]
    result_path = os.path.join(f'{RESULTS_DIR}/base-{model_name}.json')

    if not os.path.exists(result_path):
        run(['python', 'search.py',
             '--num-programs', str(n_eval_programs),
             '--output', result_path,
             '--model', base_model] + MAYBE_LOCALIZED)
        kill_dafny()

    print_done(result_path)

def run_vfp_finetuning_experiment(
    n_eval_programs: int,
    base_model: str
):
    """Run an experiment with fine-tuning on VFP."""
    model_name = base_model.split('/')[-1]
    result_path = os.path.join(f'{RESULTS_DIR}/vfp{"-sketch" if VFP_SKETCH else ""}-finetuned-{model_name}.json')
    training_set_path = f'data/vfp{"" if not VFP_SKETCH else "_sketch"}{"" if not VFP_MODULAR else "_modular"}.json'
    training_set = [training_set_path]*3 # overfitting
    model_path = f'{MODELS_DIR}/vfp{"-sketch" if VFP_SKETCH else ""}-finetuned_{model_name}'
    if not os.path.exists(result_path):
        if not os.path.exists(model_path):
            # 1- Fine-tune
            run(['python', 'training.py',
                 '--finetune',
                 '--model', base_model,
                 '--training-set', *training_set,
                 '--output', model_path,
                 ] + MAYBE_LOCALIZED + MAYBE_SKETCH)

        kill_dafny()

    print_done(result_path)

def run_dafnybench_finetuning_experiment(
    n_eval_programs: int,
    base_model: str,
    finetuning_fraction: float,
    include_graph: Optional[str] = None,
):
    """Run an experiment with fine-tuning on DafnyBench + (optionally) synthetic data.

    Args:
        n_eval_programs (int): Number of programs to hold out for the test set.
        base_model (str): Model string.
        finetuning_fraction (float): How much of the training set to use (between 0 and 1).
        include_graph (Optional[str]): If not None, then should be a path to a JSON
                                       representing an edit graph, from which we'll extract
                                       fine-tuning examples.
    """
    DAFNYBENCH_SIZE = 1326  # TODO: get this from the dataset.
    available_training_set = DAFNYBENCH_SIZE - n_eval_programs
    used_training_set = int(finetuning_fraction * available_training_set)
    skipped_training_set = available_training_set - used_training_set
    n_skip = skipped_training_set + n_eval_programs

    model_name = base_model.split('/')[-1]
    ft_percent = int(100 * finetuning_fraction)
    suffix = '+graph' if include_graph else ''

    result_path = os.path.join(
        f'{RESULTS_DIR}/finetuned-{model_name}-db{ft_percent}{suffix}.json')  # noqa

    training_set_path = f'data/finetuning_examples_{finetuning_fraction}.json'
    model_path = f'{MODELS_DIR}/finetuned_{model_name}_db{ft_percent}{suffix}'

    if not os.path.exists(result_path):
        if not os.path.exists(model_path):
            # 1- Collect training set
            run(['python', 'training.py',
                 '--extract-direct',
                 '--skip', str(n_skip),
                 '--output', training_set_path,
                 ] + MAYBE_LOCALIZED)

            training_set = [training_set_path]

            # 2- (Optional) if include_graph is provided, also include
            # examples extracted from the graph in the training set.
            if include_graph:
                graph_examples = os.path.splitext(include_graph)[0] + '-examples.json'
                run(['python', 'training.py',
                    '--extract-direct-from-graph',
                    '--graph', include_graph,
                    '--output', graph_examples,
                    ] + MAYBE_LOCALIZED)
                training_set.append(graph_examples)

            # 3- Fine-tune
            run(['python', 'training.py',
                 '--finetune',
                 '--model', base_model,
                 '--training-set', *training_set,
                 '--output', model_path,
                 ] + MAYBE_LOCALIZED)

        # 4- Evaluate
        run(['python', 'search.py',
             '--num-programs', str(n_eval_programs),
             '--output', result_path,
             '--model', model_path,
             ] + MAYBE_LOCALIZED)
        kill_dafny()

    print_done(result_path)

BASE_MODELS = [
    #'meta-llama/Meta-Llama-3.1-8B',
    #'meta-llama/CodeLlama-7b-hf'
    #'google/gemma-3-12b-it'
    #'Qwen/Qwen3-Coder-30B-A3B-Instruct'
    'Qwen/Qwen3-4B-Instruct-2507'
    #"deepseek-ai/DeepSeek-Coder-V2-Instruct"
    #'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
]

def main():
    N_EVAL_PROGRAMS = 326

    # Make huggingface tokenizers behave well with multiprocessing.
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    for m in BASE_MODELS:
        run_base_model_experiment(N_EVAL_PROGRAMS, m)

    TRAINING_SET_FRACTIONS = [.25, .5, 1.0]

    for graph in [None, 'edit_graph.json']:
        for m in BASE_MODELS:
            for f in TRAINING_SET_FRACTIONS:
                run_dafnybench_finetuning_experiment(N_EVAL_PROGRAMS, m, f, include_graph=graph)

def main_vfp():
    N_EVAL_PROGRAMS = 10
    for m in BASE_MODELS:
        run_vfp_finetuning_experiment(N_EVAL_PROGRAMS, m)

if __name__ == '__main__':
    if VFP_PROMPT:
        main_vfp()
    else:
        main()

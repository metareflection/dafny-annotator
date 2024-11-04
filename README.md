# dafny-annotator: AI-assisted verification of Dafny Programs

dafny-annotator is a tool that uses Large Language Models (LLMs) to automatically add logical annotations to Dafny programs until they verify. The project has two main components:

1. An annotation system that uses LLMs to add logical annotations (assertions, invariants, and decreases clauses) to Dafny methods. The methods are assumed to be correctly implemented and formally specified (with pre/post conditions), but may be missing annotations needed to pass Dafny's verifier.

2. A synthetic training data generation method (*DafnySynth*) based on an extensible Editor architecture. This system uses LLMs to propose new high-level ideas for Dafny programs, implement a small version of them, and incrementally edit them to create a diverse dataset of new Dafny programs for training the annotator on.

These two components are described below, as well as the experiment driver used to replicate our end-to-end evaluations with Llama 3.1 8B and CodeLlama 7B.

To start, first create an environment (virtualenv or conda) and run `pip install -r requirements.txt`. You should also install synchromesh in your environment: clone [the repository](https://github.com/kanishkg/synchromesh/), and install it locally with `python setup.py install`.

You should also clone the [DafnyBench repository](https://github.com/ChloeL19/DafnyBench/), which is the main training and evaluation set we use.

If you want to use the Llama models, you'll need to [request access on Huggingface](https://huggingface.co/meta-llama/Llama-3.1-8B). This usually takes less than an hour to be approved. Finally, make a [Weights and Biases](https://wandb.ai/) account, and run `wandb login`, if you want to use the default experiment tracking (you might need to disable it manually otherwise, since the code will currently log results to wandb).

# Annotator

The *Annotator* is the component that uses an LLM and a simple search algorithm to annotate Dafny programs. Its main components are those below: 

## program.py -- Basic Program Representation

Provides the fundamental `DafnyProgram` class with methods to:
- Call Dafny and interpret verification results (in parallel if you have a batch of programs)
- Insert/remove annotations
- Extract language modeling training examples from annotated programs

## search.py -- LLM-guided Search for Program Annotation

This is the entry point for the actual annotator. This file implements the core greedy search algorithm that uses an LLM to propose annotations, and tries to insert them at all valid program points, calling Dafny to validate the program after these insertions. Search continues until the program verifies or a maximum number of iterations is reached.

Running this script will evaluate an LLM on this annotation task. For example, if you want to evaluate the base Llama 3.1 8B model, you'd run:

```bash
python search.py \
  --num-programs 50 \
  --model meta-llama/Meta-Llama-3.1-8B \
  --output results/base-llama-8b.json \
  --benchmark-path DafnyBench/programs
```

## training.py -- Training Data Generation and Fine-tuning

This file handles two main tasks:

1. Extracting training examples from existing fully-annotated programs
2. Fine-tuning LLMs on the extracted examples

Examples:

```bash
# Extract direct prediction examples
python training.py --extract-direct \
  --skip 200 \  # Reserve those first 200 programs for evaluation
  --output data/finetuning_examples.json

# Fine-tune a model
python training.py --finetune \
  --model meta-llama/Meta-Llama-3.1-8B \
  --training-data data/finetuning_examples.json \
  --output models/finetuned_llama_8b
```

# DafnySynth: Open-ended Training Data Synthesis

DafnySynth is an extensible architecture for generating diverse training data through incremental changes to programs. It is implemented in the files we describe below.

## edit_graph.py - Edit Graph Representation

The basic structure in DafnySynth is the *Edit Graph*. This file realizes it in a directed graph structure where:

- Nodes represent either program ideas (in natural language) or concrete Dafny programs
- Edges capture the creation of programs from existing ones (or from "ideas") through various editing operations
- The graph starts from an empty root node and grows through the application of Editors

## editor.py - Editor Implementation

Editors take existing nodes and make new ones. You can extend the pipeline by creating new editors. This file defines the base Editor interface and implements several specialized editors:

- `IdeaProposer`: Generates high-level ideas for Dafny programs
- `LLMImplementer`: Converts natural language ideas into Dafny implementations, using LLMs (our experiments used GPT-4o)
- `LLMEditor`: Proposes modifications to existing programs (e.g., implementing new methods, or changing what is already there)
- `OpenAILLMAnnotator`: Adds logical annotations to partially verified programs using an OpenAI model

## synthesize_graph.py -- Incremental Program Generation

This script orchestrates the program generation process by applying a sequence of editors according to a schedule. Example usage:

```bash
# Generate a set of new programs using an initial pipeline. You can open schedules/init.json to see what editors run in what order.
python synthesize_graph.py \
  --graph programs.json \
  --schedule schedules/init.json \
  --iterations 3

# Try to verify existing programs using gpt-4o-mini
python synthesize_graph.py \
  --graph programs.json \
  --schedule schedules/verify-4omini.json \
  --iterations 1
```

The `schedules` directory has two examples of "editor schedules" (basically a sequence of editors to operate on the current graph):
- `init.json`: Full pipeline that proposes ideas, implements them, performs several rounds of edits, and adds annotations
- `verify-4omini.json`: Focused pipeline that only tries to annotate existing programs using GPT-4o mini

# Running Experiments

The `run_experiments.py` script provides a streamlined way to run comparative experiments with different models and training sets. To run our full set of experiments, you just need to run:

```bash
python run_experiments.py
```

Currently, this will:
1. Run a baseline evaluation with Llama-3 8B
2. Fine-tune the model on different fractions (25%, 50%, 100%) of DafnyBench, with and without data generated by DafnySynth
3. Evaluate each fine-tuned model

Our current findings indicate that the best results are achieved by combining DafnyBench and DafnySynth for training Llama-3 8B, suggesting that our synthetically generated programs can serve to significantly improve our annotator.

# dafny-annotator

## search.py

This file contains a simple evaluation of a "proposer" LLM guiding greedy search to annotate a Dafny program. 
To run, first clone DafnyBench here (so that DafnyBench/programs contains the benchmarks). Then, you can run with:

```sh
python search.py
```

to run the default evaluation. By default, this will use few-shot CodeLlama. You can switch between with/without rationales by changing the option that is passed to VLLMProposer (we should make it an option to cmdline.py). You can add logging in various places to get a better sense of whathappens internally (e.g. which proposals are made, which are accepted by Dafny, etc). By default it will print once it manages to verify a program, and how many programs did it make progress on (i.e. proposed some annotation that Dafny accepted).

## finetuning.py

This has several utilities to extract fine-tuning data from the benchmarks, or rationalize existing annotations. For now, fine-tuning is using either torchtune or huggingface trl. More here soon as we figure out the pipeline that works with the existing tools.

## example.py

To run example.py:

1- Clone the synchromesh repo (https://github.com/kanishkg/synchromesh/). Install locally with `python setup.py install` (with --user to install in your home, sudo to install globally, or in a virtual env)
2- Run `python example.py --model gp2`

It should download the GPT-2 weights and run it on a simple example.

GPT-2 is already bad and the prompt doesn't say anything, so it is extremely uninformed. It should print something like:

``` text
gpt2 prediction: // On line 0, add assert 0x1e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e;
gpt2 prediction: // On line 1, add assert 1a2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e;
gpt2 prediction: // On line 3, add assert 3ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd;
gpt2 prediction: // On line 1, add assert 1e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e;
gpt2 prediction: // On line 2, add assert to::interspersed[i]Interspersed[i]Interspersed[i]Interspersed[i]Interspersed[i]S;
gpt2 prediction: // On line 1, add assert 1e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e;
gpt2 prediction: // On line 1, add assert 1;
gpt2 prediction: // On line 2, add assert to::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::;
gpt2 prediction: // On line 2, add invariant is::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::;
gpt2 prediction: // On line 2, add assert to::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::;
```

Here, Synchromesh is forcing it to follow the basic format, but not the syntax of expressions, so it outputs a valid command except that the expressions wouldn't parse. We also constrain the length of the expression to at most 80 characters, since otherwise GPT-2 loops forever. The syntax issue can be fixed by using a simple Lark grammar that constrains it to valid Dafny expressions.

Before that, we should likely try a better model, like CodeLlama, with a more descriptive prompt and one or two examples, and see the predictions it makes for this example.

import regex

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from synchromesh import LarkCompletionEngine, HuggingFaceModel, predict_constrained, OpenAIModel

from cmdline import args
from completion import DafnyActionCompletionEngine


def test_dafny_completion_engine():
    from test_example import program, verification_prompt

    comp_engine = DafnyActionCompletionEngine(program)

    # Can be any huggingface model string or local path to weights.
    HF_MODEL = args.model
    gpt = AutoModelForCausalLM.from_pretrained(HF_MODEL, device_map='auto', load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    # These should work too:
    # m = RandomLanguageModel()
    # lm = OpenAIModel(model="text-curie-001", prompt_template=verification_prompt, temperature=1.)
    # Note that OpenAI now considers the Completions API as "legacy", which we use for their models.

    lm = HuggingFaceModel(gpt, tokenizer=tokenizer, prompt_template=verification_prompt, temperature=1)

    num_samples = 10
    for i in range(num_samples):
        print(HF_MODEL, "prediction:",
              predict_constrained(comp_engine, lm, 10, True, stop_tokens=["\n"]).strip())


if __name__ == '__main__':
    test_dafny_completion_engine()

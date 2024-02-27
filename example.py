import regex

from transformers import AutoTokenizer, AutoModelForCausalLM
from synchromesh import LarkCompletionEngine, HuggingFaceModel, predict_constrained, OpenAIModel

from cmdline import args

class DafnyActionCompletionEngine:
    def __init__(self, current_program):
        self._current_program_lines = current_program.split('\n')

    def complete(self, prefix: str) -> list[regex.regex]:
        # Will allow:
        # '' -> "// On line <L>, add assert"
        #     | "// On line <L>, add invariant"
        if not prefix:
            line_numbers = ('(' +
                            '|'.join(map(str, range(len(self._current_program_lines)))) +
                            ')')
            return regex.compile(f'// On line {line_numbers}, add (assert|invariant) ')
            # return regex for choosing line and action type
        else:
            # Any of the allowed characters, up to 80 of them.
            # Here, we can limit it to syntactically valid expressions
            # that also only use valid variable names.
            # Since the prefix is in the line number, we can
            # limit it to variables that have been declared
            # up to the current point.
            return regex.compile('[()><%0-9a-zA-Z\\[\\]:]{0,80};\n')

    def is_complete(self, prefix: str) -> bool:
        return prefix.endswith(';\n')


def test_dafny_completion_engine():
    from test_example import program, verification_prompt

    comp_engine = DafnyActionCompletionEngine(program)

    # Can be any huggingface model string or local path to weights.
    HF_MODEL = args.model
    gpt = AutoModelForCausalLM.from_pretrained(HF_MODEL, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    # These should work too:
    # m = RandomLanguageModel()
    # lm = OpenAIModel(model="text-curie-001", prompt_template=verification_prompt, temperature=1.)
    # Note that OpenAI now considers the Completions API as "legacy", which we use for their models.

    lm = HuggingFaceModel(gpt, tokenizer=tokenizer, prompt_template=verification_prompt, temperature=0.25)

    num_samples = 10
    for i in range(num_samples):
        print(HF_MODEL, "prediction:",
              predict_constrained(comp_engine, lm, 1, True, stop_tokens=["\n"]).strip())


if __name__ == '__main__':
    test_dafny_completion_engine()

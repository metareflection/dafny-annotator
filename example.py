import regex

from transformers import AutoTokenizer, AutoModelForCausalLM
from synchromesh import LarkCompletionEngine, HuggingFaceModel, predict_constrained, OpenAIModel


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
    program = '''
method intersperse(numbers: seq<int>, delimiter: int) returns (interspersed: seq<int>)
    ensures |interspersed| == if |numbers| > 0 then 2 * |numbers| - 1 else 0
    ensures forall i :: 0 <= i < |interspersed| ==> i % 2 == 0 ==>
                interspersed[i] == numbers[i / 2]
    ensures forall i :: 0 <= i < |interspersed| ==> i % 2 == 1 ==>
                interspersed[i] == delimiter
{
    interspersed := [];
    for i := 0 to |numbers|
    {
        if i > 0 {
            interspersed := interspersed + [delimiter];
        }
        interspersed := interspersed + [numbers[i]];
    }
}'''.strip()
    verification_prompt = f"""Given the following Dafny program:
    {program}
    Add an assertion or invariant in order to verify the program.\n"""

    comp_engine = DafnyActionCompletionEngine(program)

    # Can be any huggingface model string or local path to weights.
    HF_MODEL = 'gpt2'
    gpt2 = AutoModelForCausalLM.from_pretrained(HF_MODEL, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    # These should work too:
    # m = RandomLanguageModel()
    # lm = OpenAIModel(model="text-curie-001", prompt_template=verification_prompt, temperature=1.)
    # Note that OpenAI now considers the Completions API as "legacy", which we use for their models.

    lm = HuggingFaceModel(gpt2, tokenizer=tokenizer, prompt_template=verification_prompt, temperature=0.25)

    num_samples = 10
    for i in range(num_samples):
        print(HF_MODEL, "prediction:",
              predict_constrained(comp_engine, lm, 1, True, stop_tokens=["\n"]).strip())


if __name__ == '__main__':
    test_dafny_completion_engine()

#!/usr/bin/env python3

import regex

END = 'END###'

RATIONALE_PATTERN = '\\s?[[^\\]]*\\]'
RATIONALE_ANNOTATION_REGEX = regex.compile(f'({END})\\n|({RATIONALE_PATTERN}\\s*(assert|invariant|decreases) )')
RATIONALE_ONLY_REGEX = regex.compile(f'({END})\\n|(\\s*(assert|invariant|decreases) )')
INVARIANT_REGEX = regex.compile('[^\\n]{0,100}\\n')

class DafnyActionCompletionEngine:
    def __init__(self, current_program, with_rationale=False):
        self._current_program_lines = current_program.split('\n')
        self._with_rationale = with_rationale

    def complete(self, prefix: str) -> regex.regex:
        last_line = prefix.split('\n')[-1]
        if not last_line:
            if self._with_rationale:
                return RATIONALE_ANNOTATION_REGEX
            else:
                return RATIONALE_ONLY_REGEX
        else:
            return INVARIANT_REGEX

    def is_complete(self, prefix: str) -> bool:
        return prefix.endswith(f'{END}\n')


def make_prompt(test_program: str, with_rationale=False, actions=None) -> str:
    PROMPT_RATIONALES = [
        '[Bound the index] ',
        '[Adapt the first ensures clause as an invariant] ',
        '[Adapt the second ensures clause as an invariant] ',
        '[Adapt the third ensures clause as an invariant] ',
    ]

    if not with_rationale:
        PROMPT_RATIONALES = [''] * 4

    return f"""Given each Dafny program, propose an assertion, invariant or decreases statement in order to verify the program.
Program 1:
method maxArray(a: array<int>) returns (m: int)
  requires a.Length >= 1
  ensures forall k :: 0 <= k < a.Length ==> m >= a[k]
  ensures exists k :: 0 <= k < a.Length && m == a[k]
{{
  m := a[0];
  var index := 1;
  while (index < a.Length)
     decreases a.Length - index
  {{
    m := if m>a[index] then  m else a[index];
    index := index + 1;
  }}
}}

Action: {PROMPT_RATIONALES[0]}invariant 0 <= index <= a.Length
{END}

Program 2:
method intersperse(numbers: seq<int>, delimiter: int) returns (interspersed: seq<int>)
    ensures |interspersed| == if |numbers| > 0 then 2 * |numbers| - 1 else 0
    ensures forall i :: 0 <= i < |interspersed| ==> i % 2 == 0 ==>
                interspersed[i] == numbers[i / 2]
    ensures forall i :: 0 <= i < |interspersed| ==> i % 2 == 1 ==>
                interspersed[i] == delimiter
{{
    interspersed := [];
    for i := 0 to |numbers|
    {{
        if i > 0 {{
            interspersed := interspersed + [delimiter];
        }}
        interspersed := interspersed + [numbers[i]];
    }}
}}

Action: {PROMPT_RATIONALES[1]}invariant |interspersed| == if i > 0 then 2 * i - 1 else 0
{PROMPT_RATIONALES[2]}invariant forall i0 :: 0 <= i0 < |interspersed| ==> i0 % 2 == 0 ==> interspersed[i0] == numbers[i0 / 2]
{PROMPT_RATIONALES[3]}invariant forall i0 :: 0 <= i0 < |interspersed| ==> i0 % 2 == 1 ==> interspersed[i0] == delimiter
{END}

Program 3:
{test_program}

Action:"""

    if not actions:
        return p

    return p + ' ' + '\n'.join(actions) + f'\n{END}\n'

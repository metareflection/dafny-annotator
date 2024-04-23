#!/usr/bin/env python3

import regex

END = 'END###'

class DafnyActionCompletionEngine:
    def __init__(self, current_program):
        self._current_program_lines = current_program.split('\n')

    def complete(self, prefix: str) -> list[regex.regex]:
        # Will allow:
        # '' -> "// On line <L>, add assert"
        #     | "// On line <L>, add invariant"
        last_line = prefix.split('\n')[-1]
        if not last_line:
            #line_numbers = ('(' +
            #                '|'.join(map(str, range(len(self._current_program_lines)))) +
            #                ')')
            return regex.compile(f'({END})\\n|(\\s?\\[[^\\]]*\\]\\s*(assert|invariant|decreases) )')
        else:
            # Any of the allowed characters, up to 100 of them.
            # Here, we can limit it to syntactically valid expressions
            # that also only use valid variable names.
            # Since the prefix is in the line number, we can
            # limit it to variables that have been declared
            # up to the current point.
            #return regex.compile('[()><%0-9a-zA-Z\\[\\]:]{0,80};\n')
            return regex.compile('[^\\n]{0,100}\\n')

    def is_complete(self, prefix: str) -> bool:
        return prefix.endswith(f'{END}\n')


def make_prompt(test_program: str) -> str:
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

Action: [Bounding the index] invariant 0 <= index <= a.Length
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

Action: [Adapt the first ensures clause as an invariant] invariant |interspersed| == if i > 0 then 2 * i - 1 else 0
[Adapt the second ensures clause as an invariant] invariant forall i0 :: 0 <= i0 < |interspersed| ==> i0 % 2 == 0 ==> interspersed[i0] == numbers[i0 / 2]
[Adapt the third ensures clause as an invariant] invariant forall i0 :: 0 <= i0 < |interspersed| ==> i0 % 2 == 1 ==> interspersed[i0] == delimiter
{END}

Program 3:
{test_program}

Action:"""

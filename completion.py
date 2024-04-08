#!/usr/bin/env python3

import regex


class DafnyActionCompletionEngine:
    def __init__(self, current_program):
        self._current_program_lines = current_program.split('\n')

    def complete(self, prefix: str) -> list[regex.regex]:
        # Will allow:
        # '' -> "// On line <L>, add assert"
        #     | "// On line <L>, add invariant"
        if not prefix:
            #line_numbers = ('(' +
            #                '|'.join(map(str, range(len(self._current_program_lines)))) +
            #                ')')
            return regex.compile(f'\\s?(assert|invariant) ')
            # return regex for choosing line and action type
        else:
            # Any of the allowed characters, up to 100 of them.
            # Here, we can limit it to syntactically valid expressions
            # that also only use valid variable names.
            # Since the prefix is in the line number, we can
            # limit it to variables that have been declared
            # up to the current point.
            #return regex.compile('[()><%0-9a-zA-Z\\[\\]:]{0,80};\n')
            return regex.compile('[^;]{0,100};\n')

    def is_complete(self, prefix: str) -> bool:
        return prefix.endswith(';\n')


def make_prompt(test_program: str) -> str:
    return f"""Given the following Dafny program, propose an assertion or an invariant in order to verify the program.

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

Action 1: invariant 0 <= index <= a.Length;

Program 2:
{test_program}

Action 2:"""

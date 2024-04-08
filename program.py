import os
from typing import List
from enum import Enum

from test_example import verification_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

import dafny


class VerificationOutcome(Enum):
    FAIL = 0
    GOAL_UNPROVEN = 1
    SUCCESS = 2


class DafnyProgram:
    def __init__(self, program: str, name: str = None):
        self.lines = program.split('\n')
        self.name = name

    def insert(self, line, content):
        new_lines = self.lines.copy()
        new_lines.insert(line + 1, content)
        return DafnyProgram('\n'.join(new_lines), self.name)

    def __str__(self):
        return '\n'.join(self.lines)

    def first_line(self):
        method_line = -1
        for i in range(len(self.lines) - 1, -1, -1):
            if 'method' in self.lines[i]:
                method_line = i
                break
    
        if method_line == -1:
            return -1
    
        for i in range(method_line + 1, len(self.lines)):
            if '{' in self.lines[i]:
                return i

    def last_line(self):
        for i in range(len(self.lines) - 1, -1, -1):
            if '}' in self.lines[i]:
                return i
        return -1

    def verify(self) -> VerificationOutcome:
        result = dafny.check(str(self))

        if '0 errors' in result['out']:
            return VerificationOutcome.SUCCESS

        # TODO: check if return status 1024 is more reliable than this.
        if 'postcondition' in result['out']:
            return VerificationOutcome.GOAL_UNPROVEN

        # TODO: check if return status 512 is more reliable than this.
        return VerificationOutcome.FAIL

    def strip_annotations(self) -> 'DafnyProgram':
        start_line = self.first_line()
        if start_line == -1:
            return DafnyProgram(str(self), self.name)

        new_lines = self.lines[:start_line + 1]

        for line in self.lines[start_line + 1:]:
            first_word = line.strip().split(' ')[0]
            if first_word not in ['assert', 'invariant']:
                new_lines.append(line)

        return DafnyProgram('\n'.join(new_lines), self.name)

"""Representation of a Dafny program."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

from enum import Enum

import dafny


class VerificationOutcome(Enum):
    """
    Result of calling Dafny on a program.

    FAIL: Dafny returned an error.
    GOAL_UNPROVEN: Dafny returned a warning about unproven goals,
                   but all annotations seem to be accepted.
    SUCCESS: Dafny returned no errors or warnings.
    """

    FAIL = 0
    GOAL_UNPROVEN = 1
    SUCCESS = 2


ANNOTATION_KEYWORDS = ['assert', 'invariant', 'decreases']


class DafnyProgram:
    """
    Represents a Dafny program.

    A program has an implicit focused method, which is the method that is under
    editing, and is at the bottom of the file.

    Methods tend to be immutable (returning a new DafnyProgram instance).
    """

    def __init__(self, program: str, name: str = None):  # noqa
        self.lines = [line for line in program.split('\n') if line.strip()]
        self.name = name

    def to_json_obj(self):
        """Return a JSON-serializable representation of this object."""
        return {
            'name': self.name,
            'program': str(self)
        }

    @staticmethod
    def from_json_obj(json):
        """Inverse of to_json_obj."""
        return DafnyProgram(json['program'], json['name'])

    def insert(self, line, content: int) -> 'DafnyProgram':
        """Insert a line of code at the given line number."""
        new_lines = self.lines.copy()
        new_lines.insert(line + 1, content)
        return DafnyProgram('\n'.join(new_lines), self.name)

    def remove_line(self, line_number: int) -> 'DafnyProgram':
        """Remove the line at the given line number."""
        new_lines = self.lines[:line_number] + self.lines[line_number + 1:]
        return DafnyProgram('\n'.join(new_lines), self.name)

    def __str__(self):
        """Return the program as a string."""
        return '\n'.join(self.lines)

    def format_method_lines(self) -> str:
        """Return the focused method as a string."""
        start_line = self.method_line()
        end_line = self.last_line() or len(self.lines) - 1

        if start_line is None:
            return str(self)

        return '\n'.join(self.lines[start_line:end_line + 1])

    def method_line(self) -> int:
        """Return the line number of the focused method declaration."""
        for i in range(len(self.lines) - 1, -1, -1):
            if 'method' in self.lines[i] or 'function' in self.lines[i]:
                return i
        return None

    def first_line(self) -> int:
        """Get the number of the first line in the focused method's body."""
        method_line = self.method_line()

        if method_line is None:
            return None

        for i in range(method_line + 1, len(self.lines)):
            if '{' in self.lines[i]:
                return i

        return None

    def last_line(self):
        """Get the number of the last line in the focused method's body."""
        for i in range(len(self.lines) - 1, -1, -1):
            if '}' in self.lines[i]:
                return i
        return None

    def verify(self) -> VerificationOutcome:
        """Call Dafny on the program and return the outcome."""
        result = dafny.check(str(self))

        if '0 errors' in result['out']:
            return VerificationOutcome.SUCCESS

        # TODO: check if return status 1024 is more reliable than this.
        if 'postcondition' in result['out']:
            return VerificationOutcome.GOAL_UNPROVEN

        # TODO: check if return status 512 is more reliable than this.
        return VerificationOutcome.FAIL

    def strip_annotations(self) -> 'DafnyProgram':
        """Remove all annotations in the focused method's body."""
        start_line = self.first_line()

        if start_line is None:
            return DafnyProgram(str(self), self.name)

        new_lines = self.lines[:start_line + 1]

        for line in self.lines[start_line + 1:]:
            first_word = line.strip().split(' ')[0]
            if first_word not in ANNOTATION_KEYWORDS:
                new_lines.append(line)

        return DafnyProgram('\n'.join(new_lines), self.name)

    def strip_first_annotation(self) -> (int, str, 'DafnyProgram'):
        """Remove the first annotation in the focused method's body."""
        start_line = self.first_line()

        if start_line is None:
            return None, None, self

        for i, line in enumerate(self.lines[start_line + 1:]):
            line = line.strip()
            first_word = line.split(' ')[0]
            if first_word in ANNOTATION_KEYWORDS:
                return (start_line + i + 1,
                        line,
                        self.remove_line(start_line + i + 1))

        return None, None, self

    def annotations(self) -> list[str]:
        """Return a list of annotations in the focused method's body."""
        start_line = self.first_line()

        if start_line is None:
            return []

        annotations = []

        for line in self.lines[start_line + 1:]:
            first_word = line.strip().split(' ')[0]
            if first_word in ANNOTATION_KEYWORDS:
                annotations.append(line)

        return annotations

    def diff_annotations(self, rhs) -> set[str]:
        """Return the annotations that are in this program but not in rhs."""
        return set(self.annotations()) - set(rhs.annotations())

    def extract_examples(self) -> list[('DafnyProgram', str)]:
        """Extract annotation prediction examples from this program."""
        examples = []

        program = self

        while True:
            line_number, annotation, program = program.strip_first_annotation()
            if line_number is None:
                break
            examples.append((program, annotation))

        return examples


def worker(q, program):
    try:
        result = program.verify()
        q.put(result)
    except Exception as e:
        q.put(e)

def verify_program_with_timeout(program: DafnyProgram, timeout: float):
    """Verify a DafnyProgram in a separate process with a timeout."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q, program))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return VerificationOutcome.FAIL
    else:
        if not q.empty():
            res = q.get()
            if isinstance(res, Exception):
                return VerificationOutcome.ERROR
            else:
                return res
        else:
            return VerificationOutcome.ERROR


def parallel_verify_batch(
    programs: list[DafnyProgram],
    timeout: float = 10,
    num_processes: int = None,
) -> list[VerificationOutcome]:
    """
    Verify a batch of DafnyProgram instances in parallel.

    Args:
        programs (list[DafnyProgram]): List of programs to verify.
        timeout (float): Timeout in seconds for each verification.
        num_processes (int): Number of processes to use.

    Returns:
        Verification outcomes corresponding to the input programs (same order).
    """
    print('Verifying batch of', len(programs), 'programs')
    print('Programs:\n', "\n\n".join([str(p) for p in programs]))
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    results = [None] * len(programs)

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        future_to_index = {
            executor.submit(verify_program_with_timeout, program, timeout): i
            for i, program in enumerate(programs)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            #try:
            result = future.result()
            results[idx] = result
            #except Exception:
            #    results[idx] = VerificationOutcome.FAIL

    print('Results:', results)
    return results

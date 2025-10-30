#!/usr/bin/env python3
"""
LLM-guided search to annotate Dafny programs.

Uses a proposer (e.g., an LLM) to generate candidate annotations, and a
simple search algorithm to verify the program by adding these annotations.

This is the entry-point for the main experiments on DafnyBench.
"""

import os
import json
import argparse
from typing import Any, Optional

from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm

from program import DafnyProgram, VerificationOutcome, parallel_verify_batch
from completion import END, CODE_HERE_MARKER, make_prompt, make_prompt_for_sketch
from annotator import load_benchmarks

VFP_SKETCH = os.environ.get('VFP_SKETCH', 'false') != 'false'

class SearchNode:
    """
    Represents a node in the search tree for program annotations.

    Attributes:
        _program (DafnyProgram): The Dafny program at this node.
        _parent_node (Optional[SearchNode]): Parent node in the search tree.
        _parent_action (Optional[Any]): Action taken in the parent to get here.
        verification_outcome (Optional[VerificationOutcome]): Feedback from Dafny.
    """

    def __init__(self, program: DafnyProgram, parent_node=None,
                 parent_action=None):
        # noqa
        self._program = program
        self._parent_node = parent_node
        self._parent_action = parent_action
        self.verification_outcome = None  # Initialize as None

    def program(self) -> DafnyProgram:
        """
        Get the Dafny program at this node.

        Returns:
            DafnyProgram: The Dafny program.
        """
        return self._program

    def enumerate_successors_for_localized_annotations(
            self,
            annotations: list[str]
    ) -> list['SearchNode']:
        """
        Generate successor nodes by inserting the given annotations at valid positions.

        Args:
            annotations (list[str]): The annotations to insert.

        Returns:
            list[SearchNode]: A list of successor nodes.
        """
        indices = [i for i,line in enumerate(self._program.lines) if line.strip()==CODE_HERE_MARKER]
        if len(indices) > 0:
            i = indices[0]
            return [SearchNode(self._program.insert(i, annotation).remove_line(i)) for annotation in annotations]
        return self.enumerate_successors_with_annotations(annotations)

    def enumerate_successors_with_annotations(
            self,
            annotations: list[str]
    ) -> list['SearchNode']:
        """
        Generate successor nodes by inserting the given annotations at valid positions.

        Args:
            annotations (list[str]): The annotations to insert.

        Returns:
            list[SearchNode]: A list of successor nodes.
        """
        successors = []

        if None in (self._program.first_line(), self._program.last_line()):
            return []

        for annotation in annotations:
            for i in range(self._program.first_line(), self._program.last_line()):
                line = self._program.lines[i]
                # Ignore invariants added outside of loops
                if 'invariant' in annotation and \
                   ('for' not in line and
                    'while' not in line and
                    'invariant' not in line):  # noqa
                    continue
                new_program = self._program.insert(i, annotation)
                successor = SearchNode(new_program, self, (i, annotation))
                successors.append(successor)
        return successors


class Proposer:
    """Base class for proposers that generate candidate annotations."""

    def propose(self, nodes: list[SearchNode]) -> list[list[str]]:
        """
        Make annotation proposals for a batch of search nodes.

        Args:
            nodes (list[SearchNode]): Batch of search nodes to annotate.

        Returns:
            list[list[str]]: One list of proposed annotations for each node.
        """
        raise NotImplementedError


class VLLMProposer(Proposer):
    """
    Proposer that uses a vLLM model to generate annotation proposals.

    Attributes:
        _llm (LLM): The language model.
        _sampling_params (SamplingParams): vLLM sampling parameters.
        _with_rationale (bool): Whether LLM predictions will include a
                                rationale first, before the annotation.
    """

    def __init__(
            self,
            model_name: str,
            tokenizer: Optional[str] = None,
            num_proposals: int = 2,
            temperature: float = 1.0,
            with_rationale: bool = False,
            localized: bool = False
    ):
        # noqa
        self._llm = LLM(model=model_name,
                        tokenizer=tokenizer,
                        max_model_len=4096,
                        tensor_parallel_size=torch.cuda.device_count())
        self._sampling_params = SamplingParams(
            n=num_proposals,
            temperature=temperature,
            stop=END,
            max_tokens=300,
        )
        self._with_rationale = with_rationale
        self._localized = localized

    def localized_programs(self, node: SearchNode) -> list[str]:
        p = str(node.program())
        if not self._localized:
            return [p]
        if CODE_HERE_MARKER in p:
            return [p]
        states = node.enumerate_successors_with_annotations([CODE_HERE_MARKER])
        return [str(state.program()) for state in states]

    def propose(self, nodes: list[SearchNode]) -> list[list[str]]:
        """
        Make proposals for a batch of search nodes.

        Args:
            nodes (list[SearchNode]): The search nodes to propose annotations for.

        Returns:
            list[list[str]]: A list of lists of proposed annotations for each node.
        """
        if VFP_SKETCH:
            prompts = [make_prompt_for_sketch(str(node.program())) for node in nodes]
        else:
            prompts = [make_prompt(program, with_rationale=self._with_rationale, localized=self._localized)
                       for node in nodes for program in self.localized_programs(node)]

        # NOTE: In the future, we can plug in Synchromesh into vLLM by
        # using a logit_processor sampling param.
        # For now, we just generate unconstrained.
        responses = self._llm.generate(prompts, self._sampling_params)

        proposals = []

        for r in responses:
            raw_proposals = [o.text for o in r.outputs]
            lines = [line.strip()
                     for p in raw_proposals
                     for line in p.split('\n')]
            lines = [line for line in lines if line and line != END]

            # Rewrite rationales as comments.
            # NOTE: should refactor this to something like a
            # prompt_utils module.
            for i in range(len(lines)):
                if lines[i].startswith('[') and ']' in lines[i]:
                    comment, annotation = lines[i][1:].split(']', 1)
                    lines[i] = f'{annotation} // {comment}'.strip()

            proposals.append(lines)
        return proposals


class SearchAlgorithm:
    """Abstract base class for batch search algorithms to annotate methods."""

    def initial_state(self, programs: list[DafnyProgram]) -> Any:
        """
        Get an initial search state for annotating the given program.

        Args:
            program (DafnyProgram): The program to annotate.

        Returns:
            Any: The initial state.
        """
        raise NotImplementedError

    def step(self, state: Any, proposals: list[str]) -> Any:
        """
        Generate successor states given a list of proposals for this node.

        Args:
            state (Any): Current state.
            proposals (list[str]): Proposed annotations.

        Returns:
            list[SearchNode]: Successor states.
        """
        raise NotImplementedError

    def is_done(self, state: Any) -> (bool, Optional[DafnyProgram]):
        """
        Return whether the search is done, and if so, the verified program.

        Args:
            state (Any): The current state.

        Returns:
            (bool, Optional[DafnyProgram]): A pair indicating if the search is
                                            done, and the verified program.
        """
        raise NotImplementedError


class GreedySearch(SearchAlgorithm):
    """
    Greedy search algorithm that selects the first successful annotation.

    In this search algorithm, the state is a single program node.
    """

    def __init__(self, localized=False):
        self._localized = localized

    def initial_state(self, program: DafnyProgram) -> SearchNode:
        """
        Initialize the search state with the given program.

        Args:
            program (DafnyProgram): The program to start searching from.

        Returns:
            SearchNode: The initial search node.
        """
        # The state in greedy search is just a single program.
        return SearchNode(program)

    def step(self, state: SearchNode, proposals: list[str]) -> SearchNode:
        """
        Return the next state in the search.

        Args:
            state (SearchNode): The current search node.
            proposals (list[str]): The proposed annotations.

        Returns:
            SearchNode: The next search node.
        """
        if self._localized:
            successors = state.enumerate_successors_for_localized_annotations(proposals)
        else:
            successors = state.enumerate_successors_with_annotations(proposals)

        if len(successors) > 100:
            successors = successors[:100]

        outcomes = parallel_verify_batch([s.program() for s in successors], timeout=10)

        # Search for a successful successor first.
        for s, outcome in zip(successors, outcomes):
            s.verification_outcome = outcome

            if outcome == VerificationOutcome.SUCCESS:
                return s

        #  If not, return the first non-failure.
        for s, outcome in zip(successors, outcomes):
            if outcome != VerificationOutcome.FAIL:
                return s

        # If we couldn't make progress at all, return the original state.
        return state

    def is_done(self, state: SearchNode) -> (bool, Optional[DafnyProgram]):
        """
        Check if the search is done.

        Args:
            state (SearchNode): The current search node.

        Returns:
            (bool, Optional[DafnyProgram]): A tuple indicating if the search is done,
                and the verified program if it is.
        """
        outcome = state.verification_outcome
        if outcome == VerificationOutcome.SUCCESS:
            return True, state.program()
        return False, None


def select_programs_to_verify(
        programs: list[DafnyProgram],
        num_programs: int,
        cache_path: str,
        max_program_length: int = 2048
) -> list[DafnyProgram]:
    """
    Select programs from the benchmark that do not already verify.

    This will first take all the first `num_programs` from the benchmark
    (the ordering is consistent across runs, so this is deterministic),
    then filter out programs that Dafny already verifies without annotations.

    Since this is called every time an experiment start, and calling Dafny
    on all the benchmark programs is expensive, we cache Dafny's feedback
    to disk.

    Args:
        programs (list[DafnyProgram]): list of Dafny programs.
        num_programs (int): Number of programs to select.
        cache_path (str): Path to the verification outcome cache.
        max_program_length (int): Limit to program size (in characters).

    Returns:
        list[DafnyProgram]: A list of programs to verify.
    """
    print('Selecting programs to verify...')
    if os.path.exists(cache_path):
        with open(cache_path) as c_in:
            outcome_cache = json.load(c_in)
    else:
        outcome_cache = {}

    benchmarks = []

    for p in tqdm(programs[:num_programs]):
        if len(str(p)) > max_program_length:
            continue
        if p.name in outcome_cache:
            # NOTE: we assume the benchmark is a valid program.
            # (i.e. Dafny would not return VerificationOutcome.FAIL).
            outcome = (VerificationOutcome.SUCCESS
                       if outcome_cache[p.name] ==
                       'VerificationOutcome.SUCCESS'
                       else VerificationOutcome.GOAL_UNPROVEN)
        else:
            outcome = p.verify()
            outcome_cache[p.name] = str(outcome)
            with open(cache_path, 'w') as c_out:
                json.dump(outcome_cache, c_out)

        if outcome != VerificationOutcome.SUCCESS:
            benchmarks.append(p)

    return benchmarks



def batch_greedy_search(programs: list[DafnyProgram],
                        proposer: Proposer,
                        max_iterations: int = 5,
                        save_results_path: Optional[str] = None,
                        localized = False
                        ) -> list[Optional[DafnyProgram]]:
    """
    Perform batch greedy search to annotate programs for verification.

    Args:
        programs (list[DafnyProgram]): List of programs to search.
        proposer (Proposer): The proposer to generate annotation proposals.
        max_iterations (int): Maximum number of iterations to perform.
        save_results_path (Optional[str]): Path to save results as JSON.

    Returns:
        list[Optional[DafnyProgram]]: List of verified programs (None if not verified).
    """
    nodes = [SearchNode(p) for p in programs]
    result = [None] * len(nodes)
    is_done = [False] * len(nodes)

    method = GreedySearch(localized=localized)

    # Initialize the state for each program
    states = nodes

    # For saving results
    all_iterations = []

    for it in range(max_iterations):
        print(f'Iteration {it+1}/{max_iterations}')

        unfinished_indices = [i for i, d in enumerate(is_done) if not d]

        if not unfinished_indices:
            break

        unfinished_nodes = [states[i] for i in unfinished_indices]
        # Batch proposals.
        proposals = proposer.propose(unfinished_nodes)
        progress = 0

        # Do step in each unfinished program.
        for idx, node_idx in enumerate(unfinished_indices):
            new_state = method.step(states[node_idx], proposals[idx])

            if new_state is not states[node_idx]:
                progress += 1

            states[node_idx] = new_state
            is_done[node_idx], p = method.is_done(states[node_idx])
            if is_done[node_idx]:
                print(p.name, 'verified!')
                print(p)
                result[node_idx] = p

        # Save the state after this iteration
        iteration_data = {}
        for i, state in enumerate(states):
            program_id = programs[i].name
            iteration_data[program_id] = {
                'state': str(state.program()),
                'success': is_done[i],
            }
        all_iterations.append(iteration_data)

        if save_results_path:
            with open(save_results_path, 'w') as f:
                json.dump(all_iterations, f, indent=4)

        print('Made progress in', progress, 'programs.')

    return result


def main():
    # noqa
    parser = argparse.ArgumentParser(
        description='Evaluate dafny-annotator.')  # noqa
    parser.add_argument('--num-programs', type=int, default=50,
                        help='Number of programs to select for verification.')
    parser.add_argument('--benchmark-path', type=str,
                        default='DafnyBench/programs',
                        help='Path to the benchmark programs.')
    parser.add_argument('--max-iterations', type=int, default=5,
                        help='Maximum number of iterations for the search.')
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the LLM model to use.')
    parser.add_argument('--tokenizer', type=str, required=False,
                        help='Name of the tokenizer to use.')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results as a JSON file.')
    parser.add_argument('--cache-path', type=str,
                        default='DafnyBench/.outcome_cache.json',
                        help='Path to the verification outcome cache.')
    parser.add_argument('--max-program-length', type=int, default=2048,
                        help='Filter out benchmark programs larger than this.')
    parser.add_argument('--localized', action='store_true', help='Localize annotations')
    args = parser.parse_args()

    programs = load_benchmarks(args.benchmark_path)
    programs = [p.strip_annotations() for p in programs]

    proposer = VLLMProposer(model_name=args.model, tokenizer=args.tokenizer, with_rationale=False, localized=args.localized)

    # Select programs to verify
    benchmarks = select_programs_to_verify(
        programs,
        num_programs=args.num_programs,
        cache_path=args.cache_path,
        max_program_length=args.max_program_length)

    results = batch_greedy_search(
        benchmarks,
        proposer,
        args.max_iterations,
        args.output,
        args.localized)

    for p, r in zip(benchmarks, results):
        print('####', p.name)
        if r is None:
            print('Failed to verify')
        else:
            print('Verified program:')
            print(str(r))

    success_count = sum(r is not None for r in results)
    print('Success rate:', success_count / len(results))


if __name__ == '__main__':
    main()

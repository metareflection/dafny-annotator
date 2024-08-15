#!/usr/bin/env python3

from typing import Any, Optional

from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm

from program import DafnyProgram, VerificationOutcome
from completion import END, make_prompt
from annotator import load_benchmarks


class SearchNode:
    def __init__(self, program: DafnyProgram, parent_node=None, parent_action=None):
        self._program = program
        self._parent_node = parent_node
        self._parent_action = parent_action

    def verification_outcome(self) -> VerificationOutcome:
        return self._program.verify()

    def program(self) -> DafnyProgram:
        return self._program

    def enumerate_successors_with_annotation(self, annotation: str) -> list['SearchNode']:
        successors = []
        for i in range(0, len(self.program.lines)):
            new_program = self.program.insert(i, annotation)
            successors.append(SearchNode(new_program, self, (i, annotation)))
        return successors


class Proposer:
    def propose(self, nodes: list[SearchNode]) -> list[list[str]]:
        'Makes proposals for a batch of search nodes.'
        raise NotImplementedError


class VLLMProposer(Proposer):
    def __init__(self, model_name: str, num_proposals: int = 2, temperature: float = 1.0,
                 with_rationale: bool = False):
        self._llm = LLM(model=model_name,
                        tensor_parallel_size=torch.cuda.device_count())
        self._sampling_params = SamplingParams(
            n=num_proposals,
            temperature=temperature,
            stop=END,
            max_tokens=300,
        )
        self._with_rationale = with_rationale

    def propose(self, nodes: list[SearchNode]) -> list[list[str]]:
        proposals = []
        prompts = [make_prompt(str(node.program()), with_rationale=self._with_rationale)
                   for node in nodes]

        # NOTE: In the future, we can plug in Synchromesh into vLLM by
        # using a logit_processor sampling param.
        # For now, we just generate unconstrained.
        responses = self._llm.generate(prompts, self._sampling_params)

        proposals = []

        for r in responses:
            raw_proposals = [o.text for o in r.outputs]
            lines = [l.strip() for p in raw_proposals for l in p.split('\n')]
            lines = [l for l in lines if l and l != END]

            # Rewrite rationales as comments.
            for i in range(len(lines)):
                if lines[i].startswith('[') and ']' in lines[i]:
                    comment, annotation = lines[i][1:].split(']', 1)
                    lines[i] = f'{annotation} // {comment}'.strip()

            proposals.append(lines)
        return proposals


class SearchAlgorithm:
    def initial_state(self, program: DafnyProgram) -> Any:
        'Returns some initial state for searching for a verification of the program.'
        raise NotImplementedError

    def step(self, state: Any, proposals: list[str]) -> Any:
        'Returns the next state in the search.'
        raise NotImplementedError

    def is_done(self, state: Any) -> (bool, Optional[DafnyProgram]):
        'Returns whether the search is done, and if so, the verified program that was found.'
        raise NotImplementedError


class GreedySearch(SearchAlgorithm):
    def initial_state(self, program: DafnyProgram) -> SearchNode:
        # The state in greedy search is just a single program.
        return SearchNode(program)

    def step(self, state: SearchNode, proposals: list[str]) -> SearchNode:
        for p in proposals:
            successors = state.enumerate_successors_with_annotations(proposals)
            outcomes = [s.verification_outcome() for s in successors]

            # Look for a fully verified program first.
            for s, o in zip(successors, outcomes):
                if o == VerificationOutcome.SUCCESS:
                    return s

            # Otherwise, look for a program that Dafny does not reject.
            for s, o in zip(successors, outcomes):
                if o != VerificationOutcome.FAIL:
                    return s

        # If we couldn't make progress, return the original program.
        return state

    def is_done(self, state: SearchNode) -> (bool, Optional[DafnyProgram]):
        outcome = state.verification_outcome()
        if outcome == VerificationOutcome.SUCCESS:
            return True, state.program()
        return False, None


def batch_greedy_search(programs: list[DafnyProgram],
                        proposer: Proposer,
                        max_iterations: int = 10,
                        ) -> list[DafnyProgram]:
    nodes = [SearchNode(p) for p in programs]
    result = [None] * len(nodes)
    is_done = [False] * len(nodes)

    method = GreedySearch()

    for it in range(max_iterations):
        print(f'Iteration {it}')

        unfinished_indices = [i for i, d in enumerate(is_done) if not d]

        if not unfinished_indices:
            break

        unfinished_nodes = [nodes[i] for i in unfinished_indices]
        proposals = proposer.propose(unfinished_nodes)
        for i in unfinished_indices:
            nodes[i] = method.step(nodes[i], proposals[i])
            is_done[i], p = method.is_done(nodes[i])
            if is_done[i]:
                result[i] = p

    return result


if __name__ == '__main__':
    from cmdline import args

    # Load the Dafny programs to be verified.
    N = 50
    programs = load_benchmarks()
    programs = [p.strip_annotations() for p in programs[:N]]

    # Get N programs that are not already verified.
    print('Selecting programs to verify...')
    benchmarks = []

    for p in tqdm(programs):
        if p.verify() != VerificationOutcome.SUCCESS:
            benchmarks.append(p)
            if len(benchmarks) == N:
                break

    proposer = VLLMProposer(model_name=args.model)
    results = batch_greedy_search(programs, proposer)

    print('Success rate:', sum(r is not None for r in results) / len(results))

    for p, r in zip(programs, result):
        print('####', p.name)

        if r is None:
            print('Failed to verify')
        else:
            print('Verified program:')
            print(str(r))

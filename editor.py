#!/usr/bin/env python3

"""
Editors take nodes in the edit graph and create new ones in batch.

Notable editors:
- IdeaProposer: takes the root node, and creates high-level idea nodes.
- LocalLLMAnnotator: takes partially verified program nodes and attempts
    to add annotations to them with a local model.
- OpenAILLMAnnotator: same, but with OpenAI models.
- LLMImplementer: takes idea nodes and tries to implement them in Dafny.
- LLMEditor: takes a program node and proposes a diff to it.
"""

import uuid
import json

from edit_graph import Node
from prompts import replace_in_prompt
from program import DafnyProgram, VerificationOutcome, parallel_verify_batch

import openai


class Editor:
    """An Editor creates new nodes in the graph from existing ones."""

    def name(self) -> str:
        """
        Return a readable identifier for the editor.

        Should generally start with the Editor class name,
        followed by any distinguishing parameters (e.g. model name).
        """
        raise NotImplementedError

    def can_edit(self, node: Node) -> bool:
        """Return whether this node is a valid candidate for this editor."""
        raise NotImplementedError

    def edit_batch(self, nodes: list[Node]) -> list[Node]:
        """
        Create new nodes based on the given nodes.

        All nodes in the input list are guaranteed to satisfy `can_edit`.
        """
        raise NotImplementedError


class IdeaProposer(Editor):
    """
    Proposes high-level ideas for new verified programs using an LLM.

    This adds nodes to the graph based only on the root node and idea nodes.
    Existing ideas are included in the prompt and we ask the model to not
    repeat them.
    """

    def __init__(
            self,
            model_str: str,
            temperature: float = 0.5,
            n_ideas: int = 50
    ):
        """Initialize an IdeaProposer based on an OpenAI model."""
        self._model_str = model_str
        self._client = openai.Client()
        self._n_ideas = n_ideas
        self._temperature = temperature
        with open('prompts/idea_proposer.json', 'r') as f:
            self._prompt_new = json.load(f)
        with open('prompts/idea_proposer_with_existing.json', 'r') as f:
            self._prompt_existing = json.load(f)

    def name(self) -> str:
        """Editor name."""
        return f'IdeaProposer-{self._model_str}'

    def can_edit(self, node: Node) -> bool:
        """Check for root or idea node."""
        return node.type in ('root', 'idea')

    def edit_batch(self, nodes: list[Node]) -> list[Node]:
        """Propose new ideas for verified programs using an LLM."""
        # Check if there are any idea nodes.
        # If so, use the prompt that includes existing ideas to try to
        # avoid repetition.
        existing_ideas = []
        if any(node.type == 'idea' for node in nodes):
            prompt = self._prompt_existing
            existing_ideas = [node.content
                              for node in nodes
                              if node.type == 'idea']
        else:
            prompt = self._prompt_new

        prompt = replace_in_prompt(prompt, '$N', str(self._n_ideas))
        prompt = replace_in_prompt(prompt,
                                   '$IDEAS',
                                   '\n'.join(sorted(existing_ideas)))

        response = self._client.chat.completions.create(
            model=self._model_str,
            messages=prompt,
            temperature=self._temperature,
        )

        ideas = response.choices[0].message.content.split('\n')
        return [Node(
            id=f'idea-{i + len(existing_ideas)}',
            type='idea',
            content=idea,
            parents=['root'],
        ) for i, idea in enumerate(ideas)]


class LLMImplementer(Editor):
    """Takes idea nodes and tries to implement them in Dafny using an LLM."""

    def __init__(self, model_str: str, temperature: float = 0.2):
        """Initialize an LLMImplementer based on an OpenAI model."""
        self._model_str = model_str
        self._temperature = temperature
        self._client = openai.Client()
        with open('prompts/llm_implementer.json', 'r') as f:
            self._prompt = json.load(f)

    def name(self) -> str:
        """Editor name."""
        return f'LLMImplementer-{self._model_str}'

    def can_edit(self, node: Node) -> bool:
        """Only edit idea nodes."""
        return node.type == 'idea'

    def edit_batch(self, nodes: list[Node]) -> list[Node]:
        """Implement ideas nodes as Dafny programs."""
        new_nodes = []
        programs = []
        parent_nodes = []

        # Generate Dafny programs from ideas using the LLM
        for node in nodes:
            prompt = replace_in_prompt(self._prompt, '$IDEA', node.content)
            response = self._client.chat.completions.create(
                model=self._model_str,
                messages=prompt,
                temperature=self._temperature,
                max_tokens=1024,
            )
            output = response.choices[0].message.content
            program_str = self._extract_dafny_program(output)
            if not program_str:
                continue
            new_program = DafnyProgram(program_str)
            programs.append(new_program)
            parent_nodes.append(node)

        outcomes = parallel_verify_batch(programs)

        for program, outcome, parent_node in zip(programs, outcomes,
                                                 parent_nodes):
            if outcome != VerificationOutcome.FAIL:
                new_node = Node(
                    id=str(uuid.uuid4()),
                    type='program',
                    content=str(program),
                    parents=[parent_node.id],
                    properties={'verification_outcome': outcome.name}
                )
                new_nodes.append(new_node)
                print(f"Program inspired by {parent_node.id} verified.")
            else:
                print(f"Program inspired by {parent_node.id} did not verify.")
        return new_nodes

    def _extract_dafny_program(self, text: str) -> str:
        """Extract the Dafny program between the markers."""
        start_marker = '// BEGIN DAFNY'
        end_marker = '// END DAFNY'
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        if start_idx == -1 or end_idx == -1:
            return None
        return text[start_idx + len(start_marker):end_idx].strip()


class LLMEditor(Editor):
    """Takes a program node and proposes a diff to it using an LLM."""

    def __init__(self, model_str: str, temperature: float = 0.2):
        """Initialize an LLMEditor based on an OpenAI model."""
        self._model_str = model_str
        self._temperature = temperature
        self._client = openai.Client()
        with open('prompts/llm_editor.json', 'r') as f:
            self._prompt = json.load(f)

    def name(self) -> str:
        """Editor name."""
        return f'LLMEditor-{self._model_str}'

    def can_edit(self, node: Node) -> bool:
        """Return True if node is a program node."""
        return node.type == 'program'

    def edit_batch(self, nodes: list[Node]) -> list[Node]:
        """Propose diffs to program nodes using parallel verification."""
        new_nodes = []
        programs = []
        parent_nodes = []
        new_program_contents = []

        # Generate new programs from diffs
        for node in nodes:
            program_lines = node.content.strip().split('\n')
            numbered_lines = '\n'.join(
                [f"{i+1}: {line}" for i, line in enumerate(program_lines)]
            )
            prompt = replace_in_prompt(self._prompt,
                                       '$PROGRAM', numbered_lines)
            response = self._client.chat.completions.create(
                model=self._model_str,
                messages=prompt,
                temperature=self._temperature,
            )
            output = response.choices[0].message.content
            new_program_str = self._apply_diff(node.content, output)
            if not new_program_str:
                continue
            new_program = DafnyProgram(new_program_str)
            programs.append(new_program)
            parent_nodes.append(node)
            new_program_contents.append(new_program_str)

        if not programs:
            return new_nodes  # No valid programs generated

        # Verify programs in parallel
        outcomes = parallel_verify_batch(programs, timeout=10)

        # Process verification outcomes
        for program_str, outcome, parent_node in zip(new_program_contents,
                                                     outcomes,
                                                     parent_nodes):
            if outcome != VerificationOutcome.FAIL:
                new_node = Node(
                    id=str(uuid.uuid4()),
                    type='program',
                    content=program_str,
                    parents=[parent_node.id],
                    properties={'verification_outcome': outcome.name}
                )
                new_nodes.append(new_node)
                print(f"Successfully edited program from {parent_node.id}.")
            else:
                print(f"Program from node {parent_node.id} did not verify.")
        return new_nodes

    def _apply_diff(self, original_program: str, diff_text: str) -> str:
        """Apply the diff from the LLM to the original program."""
        start_marker = '// BEGIN DAFNY AT LINE'
        end_marker = '// END DAFNY'
        start_idx = diff_text.find(start_marker)
        end_idx = diff_text.find(end_marker)
        if start_idx == -1 or end_idx == -1:
            return None
        try:
            line_info = diff_text[start_idx:start_idx + len(start_marker) + 10]
            N = int(line_info.split(start_marker)[1].strip())
        except ValueError:
            N = None
        new_lines = (diff_text[start_idx:end_idx]
                     .strip().split('\n')[1:])  # Skip initial comment.
        original_lines = original_program.strip().split('\n')
        if N is None or N > len(original_lines):
            N = len(original_lines)
        new_program_lines = original_lines[:N-1] + new_lines
        return LLMEditor._close_braces('\n'.join(new_program_lines))

    @staticmethod
    def _close_braces(program: str) -> str:
        brace_count = max(0, program.count('{') - program.count('}'))
        return program + '\n}' * brace_count


class OpenAILLMAnnotator(Editor):
    """
    Adds logical annotations to a program failing to verify.

    Takes a program node whose verification outcome is GOAL_UNPROVEN,
    and calls an LLM using the OpenAI API to try to get Dafny to verify it.
    """

    def __init__(self, model_str: str, temperature: float = 0.2):
        """Initialize an OpenAILLMAnnotator based on an OpenAI model."""
        self._model_str = model_str
        self._temperature = temperature
        self._client = openai.Client()
        with open('prompts/openai_llm_annotator.json', 'r') as f:
            self._prompt = json.load(f)

    def name(self) -> str:
        """Editor name."""
        return f'OpenAILLMAnnotator-{self._model_str}'

    def can_edit(self, node: Node) -> bool:
        """Return True for program nodes that need verification annotations."""
        return (node.type == 'program' and
                node.properties.get('verification_outcome') == 'GOAL_UNPROVEN')

    def edit_batch(self, nodes: list[Node]) -> list[Node]:
        """Add annotations to programs to try to make them verify."""
        new_nodes = []
        programs = []
        parent_nodes = []
        new_program_contents = []

        # Generate modified programs from annotations
        for node in nodes:
            prompt = replace_in_prompt(self._prompt, '$PROGRAM', node.content)
            response = self._client.chat.completions.create(
                model=self._model_str,
                messages=prompt,
                temperature=self._temperature,
                max_tokens=1024,
            )
            output = response.choices[0].message.content
            program_str = self.extract_modified_program(output)
            if not program_str:
                continue
            new_program = DafnyProgram(program_str)
            programs.append(new_program)
            parent_nodes.append(node)
            new_program_contents.append(program_str)

        if not programs:
            return new_nodes  # No valid programs generated

        # Verify programs in parallel
        outcomes = parallel_verify_batch(programs, timeout=60)

        # Process verification outcomes
        for program_str, outcome, parent_node in zip(new_program_contents,
                                                     outcomes,
                                                     parent_nodes):
            if outcome != VerificationOutcome.FAIL:
                print("Dafny feedback after annotations:", str(outcome))
                new_node = Node(
                    id=str(uuid.uuid4()),
                    type='program',
                    content=program_str,
                    parents=[parent_node.id],
                    properties={'verification_outcome': outcome.name}
                )
                new_nodes.append(new_node)
                print(f"Successfully verified program from node {parent_node.id}.")
            else:
                print("Program fails after attempt to annotations.")
        return new_nodes

    def extract_modified_program(self, text: str) -> str:
        """Extract the modified Dafny program between the markers."""
        start_marker = '// BEGIN MODIFIED PROGRAM'
        end_marker = '// END MODIFIED PROGRAM'
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        if start_idx == -1 or end_idx == -1:
            return None
        return text[start_idx + len(start_marker):end_idx].strip()

#!/usr/bin/env python3

"""
Selection policies for choosing which nodes to edit in each iteration.

A selection policy is called at each iteration of the graph creation
procedure to filter a subset of nodes to edit. The simplest selection
is a random selection, but better policies can be used to balance
exploration and exploitation: focusing on nodes that have been more
productive in terms of finding large programs with many invariants,
while still revisiting underexplored ideas (but avoiding nodes that
have been attempted many times and failed).

TODO: design something like an UCTPolicy.
"""

import random
from abc import ABC, abstractmethod

from edit_graph import Node
from editor import Editor


class SelectionPolicy(ABC):
    """Abstract base class for node selection policies."""

    @abstractmethod
    def select_batch(self, nodes: list[Node], editor: Editor) -> list[Node]:
        """
        Select a subset of nodes to edit from the given list.

        Args:
            nodes (list[Node]): List of available nodes.
            editor (Editor): The editor that will be applied.

        Returns:
            list[Node]: The selected subset of nodes.
        """
        raise NotImplementedError

    def process_new(self, new_nodes: list[Node]) -> None:
        """
        Process the newly created nodes from the last iteration.

        This allows the selection policy to (optionally)
        compute relevant statistics, such as visit counts in MCTS.

        Args:
            new_nodes (list[Node]): List of nodes created by the Editor that
                                    ran last.
        """
        pass


class RandomSelection(SelectionPolicy):
    """Randomly selects a maximum number of nodes from the current batch."""

    def __init__(self, max_nodes: int):
        """
        Initialize a RandomSelection policy.

        Args:
            max_nodes (int): Maximum number of nodes to select in each batch.
        """
        self._max_nodes = max_nodes

    def select_batch(self, nodes: list[Node], editor: Editor) -> list[Node]:
        """Select a random subset of nodes from the given list."""
        return random.sample(nodes, min(self._max_nodes, len(nodes)))


class SelectAll(SelectionPolicy):
    """Trivially selects all nodes in the batch."""

    def select_batch(self, nodes: list[Node], editor: Editor) -> list[Node]:
        """Select all nodes."""
        return nodes

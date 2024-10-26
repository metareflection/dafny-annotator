"""
A directed graph of nodes capturing evolving Dafny programs.

A graph starts with a single "root" node, and it grows by the effect
of "Editors". See editors.py for the editors that create new nodes
out of existing ones.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Any
import uuid


class NodeType(Enum):
    """Types of Nodes in the Edit Graph."""

    ROOT = 'root'
    IDEA = 'idea'
    PROGRAM = 'program'


@dataclass
class Node:
    """
    A node in the "Edit Graph", which represent evolving Dafny programs.

    There are several kinds of nodes:
    - "root": Initial node of the graph, containing no information.
    - "idea": A node representing a high-level idea for a program,
       in natural language.
    - "program": A node representing a complete Dafny program.
    """

    id: str
    type: str  # NodeType
    content: str
    # List of parent node IDs
    parents: list[str] = field(default_factory=list)
    # Additional statistics (e.g., visit count): used during node selection.
    statistics: dict[str, Union[int, float]] = field(default_factory=dict)
    # Editors can save additional data here.
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Assign a unique ID to the node if it doesn't have one."""
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class EditGraph:
    """
    A directed graph of nodes representing the evolution of a program idea.

    Contains nodes, and each node contains edges to its parents.
    """

    nodes: dict[str, Node] = field(default_factory=dict)

    @staticmethod
    def create() -> 'EditGraph':
        """Create an EditGraph with a single root node."""
        root = Node(id='root', type=NodeType.ROOT.value, content='')
        return EditGraph(nodes={'root': root})

    @staticmethod
    def load_graph(graph_input_path: str) -> 'EditGraph':
        """Load an EditGraph from a JSON file."""
        with open(graph_input_path, 'r') as f:
            graph_data = json.load(f)

        nodes = {}

        for node_data in graph_data['nodes']:
            node = Node(
                id=node_data['id'],
                type=node_data['type'],
                content=node_data['content'],
                parents=node_data.get('parents', []),
                statistics=node_data.get('statistics', {}),
                properties=node_data.get('properties', {})
            )
            nodes[node.id] = node
        return EditGraph(nodes)

    def save(self, graph_output_path: str):
        """Dump the graph to a JSON file."""
        graph_data = {'nodes': []}
        for node in self.nodes.values():
            node_data = {
                'id': node.id,
                'type': node.type,
                'content': node.content,
                'parents': node.parents,
                'statistics': node.statistics,
                'properties': node.properties
            }
            graph_data['nodes'].append(node_data)

        with open(graph_output_path, 'w') as f:
            json.dump(graph_data, f, indent=4)

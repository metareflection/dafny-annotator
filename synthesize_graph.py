#!/usr/bin/env python3
"""Run a pipeline to create or evolve a graph of Dafny programs."""

import argparse
import os
import json

from edit_graph import EditGraph
from editor import Editor
from editor import *  # noqa
from selection import SelectionPolicy, RandomSelection, SelectAll


def run_edititing_iterations(
    graph: EditGraph,
    editors: list[Editor],
    selection_policy: SelectionPolicy,
    num_iterations: int,
    graph_output_path: str,
):
    """
    Run editing iterations over the graph using editors and a selection policy.

    Args:
        graph (EditGraph): The edit graph.
        editors (list[Editor]): List of editors to apply.
        selection_policy (SelectionPolicy): The batch selection policy.
        num_iterations (int): Number of iterations to run.
        graph_output_path (str): Path to save the graph after each iteration.

    This will edit the graph in-place, by creating new nodes in batches.
    The graph will be saved to `graph_output_path` after each iteration.
    """
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        for editor in editors:
            print(f"Applying editor: {editor.name()}")

            # Collect all nodes that can be edited by this editor
            available_nodes = [
                node for node in graph.nodes.values() if editor.can_edit(node)
            ]

            # Use selection policy to select nodes
            selected_nodes = selection_policy.select_batch(available_nodes,
                                                           editor)
            print(f"Selected nodes for editing: {len(selected_nodes)}")

            if not selected_nodes:
                print("No nodes selected for editing.")
                continue

            # Editor creates new nodes
            new_nodes = editor.edit_batch(selected_nodes)
            print(f"New nodes created: {len(new_nodes)}")

            # Add new nodes to graph
            for node in new_nodes:
                graph.nodes[node.id] = node

            # Let selection policy process new nodes
            selection_policy.process_new(new_nodes)

            # Save the graph after each editor's application
            graph.save(graph_output_path)
            print(f"Graph saved to {graph_output_path}")


def load_schedule(schedule_path: str):
    """
    Load a "schedule" (a sequence of editors) from a JSON file.

    Args:
        schedule_path (str): Path to the schedule JSON file.

    Returns:
        list[Editor]: A list of instantiated editors with parameters given
                      in the JSON.
    """
    with open(schedule_path, 'r') as f:
        schedule_data = json.load(f)

    editors = []
    for editor_spec in schedule_data.get('editors', []):
        editor_class_name = editor_spec['editor_class']
        parameters = editor_spec.get('parameters', {})

        editor_class = globals().get(editor_class_name)
        if editor_class is None:
            raise ValueError(f"No Editor class '{editor_class_name}'.")

        editor = editor_class(**parameters)
        editors.append(editor)

    return editors


def main():  # noqa
    parser = argparse.ArgumentParser(
        description="Run iterations of a sequence of graph editing pipeline.")
    parser.add_argument('--graph', type=str, required=True,
                        help='Path to the graph JSON file.')
    parser.add_argument('--schedule', type=str, required=True,
                        help='Path to the schedule JSON file.')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of iterations to run.')
    parser.add_argument('--skip', type=int, default=0,
                        help='Number of editors from the schedule to skip.')
    parser.add_argument('--selection-policy', type=str,
                        choices=['RandomSelection', 'SelectAll'],
                        default='RandomSelection',
                        help='Selection policy to use.')
    parser.add_argument('--max-nodes', type=int, default=5,
                        help='Maximum number of nodes to select.')

    args = parser.parse_args()

    if os.path.exists(args.graph):
        print(f"Loading graph from {args.graph}")
        graph = EditGraph.load_graph(args.graph)
    else:
        print("Initializing new graph.")
        graph = EditGraph.create()

    print(f"Loading Editors schedule from {args.schedule}")
    editors = load_schedule(args.schedule)[args.skip:]

    # Instantiate the selection policy
    if args.selection_policy == 'RandomSelection':
        selection_policy = RandomSelection(max_nodes=args.max_nodes)
    else:
        selection_policy = SelectAll()

    # Run the iterations
    run_edititing_iterations(
        graph=graph,
        editors=editors,
        selection_policy=selection_policy,
        num_iterations=args.iterations,
        graph_output_path=args.graph
    )


if __name__ == '__main__':
    main()

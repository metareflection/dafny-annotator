#!/usr/bin/env python3
"""Visualize the Edit Graph."""

import argparse
from collections import deque

import networkx as nx
from graphviz import Digraph

from edit_graph import EditGraph


def visualize_graph(graph: EditGraph, output_path: str):  # noqa
    """
    Visualize the program edit graph using GraphViz.

    Args:
        graph (EditGraph): The edit graph to visualize.
        output_path (str): Path to save the GraphViz DOT file.
    """
    edges = []
    for node in graph.nodes.values():
        node_id = node.id
        parents = node.parents
        for parent_id in parents:
            edges.append((parent_id, node_id))

    G = nx.DiGraph()
    for node_id, node in graph.nodes.items():
        G.add_node(node_id, data=node)
    G.add_edges_from(edges)

    # Initialize distances
    for node_id in G.nodes():
        G.nodes[node_id]['distance'] = None

    idea_nodes = [
        node_id for node_id, data in G.nodes(data=True)
        if data['data'].type == 'idea'
    ]
    max_distance = 0

    # BFS to compute distances from idea nodes
    for idea_node in idea_nodes:
        queue = deque()
        queue.append((idea_node, 0))
        while queue:
            current_node, dist = queue.popleft()
            node_data = G.nodes[current_node]
            # Update distance if it's None or greater than current dist
            if node_data['distance'] is None or node_data['distance'] > dist:
                node_data['distance'] = dist
                max_distance = max(max_distance, dist)
                # Enqueue neighbors
                for neighbor in G.successors(current_node):
                    queue.append((neighbor, dist + 1))

    # Define colors for gradient of program nodes
    light_blue_rgb = (170, 210, 230)  # Light blue
    dark_blue_rgb = (0, 0, 160)       # Dark blue

    def interpolate_color(c1: tuple, c2: tuple, t: float) -> tuple:
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )

    def rgb_to_hex(rgb: tuple) -> str:
        return '#%02x%02x%02x' % rgb

    dot = Digraph(comment='Dafny Program Graph')
    dot.attr('node', style='filled', fontname='Helvetica')

    for node_id, data in G.nodes(data=True):
        node = data['data']
        label = node_id
        node_attrs = {'label': label}
        node_type = node.type

        if node_type == 'idea':
            # Idea nodes: yellow
            node_attrs['fillcolor'] = 'yellow'
            node_attrs['shape'] = 'ellipse'
            node_attrs['fontsize'] = '16'
            node_attrs['width'] = '1'
            node_attrs['height'] = '1'
            node_attrs['fixedsize'] = 'true'

        elif node_type == 'program':
            # Program nodes: gradient color based on distance
            dist = data['distance'] if data['distance'] is not None else 0
            t = dist / max(1, max_distance)
            color_rgb = interpolate_color(light_blue_rgb, dark_blue_rgb, t)
            color_hex = rgb_to_hex(color_rgb)
            node_attrs['fillcolor'] = color_hex
            node_attrs['shape'] = 'box'
            node_attrs['fontsize'] = '16'
            # Dark text on light background, light text on dark background
            node_attrs['fontcolor'] = rgb_to_hex(
                interpolate_color((100, 100, 100), (255, 255, 255), t)
            )
            # Compute node size based on number of lines
            content = node.content
            num_lines = len(content.strip().split('\n'))
            base_size = 0.5
            size = base_size + num_lines**0.6 * 0.2
            node_attrs['width'] = str(size)
            node_attrs['height'] = str(size)
            node_attrs['fixedsize'] = 'true'

        elif node_type == 'root':
            # Root node: grey circle
            node_attrs['fillcolor'] = 'lightgrey'
            node_attrs['shape'] = 'circle'
            node_attrs['fontsize'] = '16'
            node_attrs['width'] = '1'
            node_attrs['height'] = '1'
            node_attrs['fixedsize'] = 'true'

        else:
            node_attrs['shape'] = 'circle'

        dot.node(node_id, **node_attrs)

    for u, v in G.edges():
        dot.edge(u, v)

    dot.render(output_path, format='png', cleanup=True)
    print(f"Graph visualization saved to {output_path}.png")


def main():  # noqa
    parser = argparse.ArgumentParser(
        description="Visualize a program edit graph.")
    parser.add_argument('--graph', type=str, required=True,
                        help='Path to the graph JSON file.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Output path (without the extension)')

    args = parser.parse_args()

    # Load or create the graph
    print(f"Loading graph from {args.graph}")
    graph = EditGraph.load_graph(args.graph)

    # Optionally visualize after running iterations
    visualize_graph(graph, args.output_path)
    print(f"Visualization saved to {args.output_path}.png")


if __name__ == '__main__':
    main()

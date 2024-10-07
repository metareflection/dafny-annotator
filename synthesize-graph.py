#!/usr/bin/env python3
# Synthesize annotated Dafny programs based on a seed set.

import argparse
import json
import random
import math
from collections import deque

import openai
from tqdm import tqdm
import networkx as nx
from graphviz import Digraph

from annotator import load_benchmarks
from program import DafnyProgram, VerificationOutcome


client = openai.Client()
benchmark = load_benchmarks('./DafnyBench/programs')


class Node:
    def __init__(self, node_id, node_type, content, parents=None, diff=None):
        self.id = node_id
        self.type = node_type  # 'idea' or 'program'
        self.content = content  # For 'idea', a string; for 'program', a DafnyProgram instance
        self.parents = parents or []  # List of parent node IDs
        self.diff = diff  # The diff applied to generate this node (if applicable)


def load_graph(graph_input_path):
    with open(graph_input_path, 'r') as f:
        graph_data = json.load(f)
    graph = {}
    for node_data in graph_data['nodes']:
        if node_data['type'] == 'program':
            content = DafnyProgram(node_data['content'])
        else:
            content = node_data['content']
        node = Node(
            node_id=node_data['id'],
            node_type=node_data['type'],
            content=content,
            parents=node_data.get('parents', []),
            diff=node_data.get('diff', None)
        )
        graph[node.id] = node
    return graph


def save_graph(graph, graph_output_path):
    graph_data = {'nodes': []}
    for node in graph.values():
        node_data = {
            'id': node.id,
            'type': node.type,
            'content': node.content if node.type == 'idea' else str(node.content),
            'parents': node.parents,
            'diff': node.diff
        }
        graph_data['nodes'].append(node_data)
    with open(graph_output_path, 'w') as f:
        json.dump(graph_data, f, indent=4)


def initialize_graph_from_ideas(ideas_input_path):
    with open(ideas_input_path, 'r') as f:
        ideas = json.load(f)
    graph = {}
    for idx, idea in enumerate(ideas):
        node_id = f'idea_{idx}'
        node = Node(node_id=node_id, node_type='idea', content=idea)
        graph[node_id] = node
    return graph


def sample_seed_nodes(graph, M):
    nodes = list(graph.values())
    return random.sample(nodes, min(M, len(nodes)))


def sample_inspiration_node(graph, seed_node):
    # Exclude the seed node
    nodes = [node for node in graph.values() if node.id != seed_node.id]
    # Prefer 'program' nodes
    program_nodes = [node for node in nodes if node.type == 'program']
    if program_nodes:
        return random.choice(program_nodes)
    else:
        # Load a random program from DafnyBench
        random_program = random.choice(benchmark)
        temp_node = Node(node_id='benchmark', node_type='program', content=random_program)
        return temp_node


def generate_prompt(seed_node, inspiration_node):
    system_message = "You are a creative and correct Dafny programmer. Your goal is to help generate novel, interesting Dafny programs to serve as diverse training data for an AI model."

    user_message = f"Here is a random Dafny program: ```\n{inspiration_node.content}\n```\n\n"

    if seed_node.type == 'idea':
        user_message += "Your goal now is to generate a new, small Dafny program. Here is an inspiration idea for you:\n"
        user_message += f"{seed_node.content}\n\n"
        user_message += """Please propose a small, complete program inspired by this idea. Your output should be a valid Dafny program starting with a line "// BEGIN DAFNY", ending with a line "// END DAFNY". The progam should be simple -- it is OK to not fully implement the idea I gave, but try to propose something related to it. To maximize the chance that the output is correct, please do not be overly ambitious (propose 1-3 methods and/or data structures, as seems fit), yet still be a bit creative. Do not include a Main() method -- think of this as an evolving project. Also, don't include testing functions - we just want functions that satisfy their formal specifications. We're most interested in getting examples of loop invariants."""
    else:
        user_message = "Your goal now is to generate a small patch to be applied to this existing Dafny program:\n"
        lines = str(seed_node.content).split('\n')
        numbered_lines = '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])
        user_message += f"{numbered_lines}\n\n"
        user_message += """Please propose a small modification to the current program based on the inspiration above. Your output should be a valid Dafny program starting with "// BEGIN DAFNY AT LINE N", where N is some line number referring to the existing program. Note that N wil be the number of your first line (i.e. current line N will be deleted -- to append, use last line + 1). On your last line, write "// END DAFNY". The modification should be simple, at most adding or changing a few methods or data structures, or modifying existing code (e.g., rewrite it in a different way, or implement something related but different). To maximize the chance that the output is correct, please do not be overly ambitious, yet still be a bit creative.  Do not include a Main() method -- think of this as an evolving project. Also, don't include testing functions - we just want functions that satisfy their formal specifications. We're most interested in getting examples of loop invariants."""

    return system_message, user_message


def close_braces(program: DafnyProgram) -> DafnyProgram:
    brace_count = 0
    for line in program.lines:
        brace_count += line.count('{')
        brace_count -= line.count('}')
    new_lines = program.lines.copy()
    new_lines.extend(['}'] * brace_count)
    return DafnyProgram('\n'.join(new_lines), program.name)


def call_llm(system_message, user_message, model, temperature=0.2, max_tokens=1024):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def parse_and_apply_diff(seed_node, response):
    response_begin = response.find('// BEGIN DAFNY')

    if response_begin == -1:
        print("Error: LLM output does not contain '// BEGIN DAFNY'")
        return None

    response_end = response.find('// END DAFNY')
    if response_end == -1:
        # Try the whole program anyway.
        response_end = len(response)

    response = response[response_begin:response_end]
    lines = response.strip().split('\n')

    if not lines:
        return None
    try:
        if 'AT LINE' in lines[0]:
            N = int(lines[0].split('// BEGIN DAFNY AT LINE')[1].strip())
        else:
            N = 0
    except ValueError:
        print("Warning: Could not parse line number N -- trying 0")
        N = 0
    new_program_lines = lines[1:]
    if seed_node.type == 'idea':
        # Seed program is empty
        new_program = '\n'.join(new_program_lines)
        return close_braces(DafnyProgram(new_program))
    elif seed_node.type == 'program':
        seed_program_lines = str(seed_node.content).split('\n')
        if N > len(seed_program_lines):
            print("Line number N exceeds seed program length -- appending")
            N = len(seed_program_lines)
        # Remove lines up to N (inclusive)
        modified_program_lines = seed_program_lines[:N-1] + new_program_lines
        new_program = '\n'.join(modified_program_lines)
        return close_braces(DafnyProgram(new_program))
    return None


def generate_new_node_id(graph):
    return f'node_{len(graph)}'


def generate_new_node(seed_node, inspiration_node, model, graph):
    system_message, user_message = generate_prompt(seed_node, inspiration_node)
    response = call_llm(system_message, user_message, model)
    new_program = parse_and_apply_diff(seed_node, response)
    if new_program:
        outcome = new_program.verify()
        if outcome != VerificationOutcome.FAIL:
            new_node_id = generate_new_node_id(graph)
            new_node = Node(
                node_id=new_node_id,
                node_type='program',
                content=new_program,
                parents=[seed_node.id],
                diff=response,
            )
            print(f"Generated new node {new_node_id} from seed node {seed_node.id}")
            return new_node
        else:
            print("New program did not verify with Dafny.")
            print('Program:')
            print(new_program)
    else:
        print("Failed to generate new program.")
    return None


def run_iterations(graph, K, M, model, graph_output_path):
    for iteration in range(K):
        print(f"Iteration {iteration+1}/{K}")
        seed_nodes = sample_seed_nodes(graph, M)
        for seed_node in seed_nodes:
            inspiration_node = sample_inspiration_node(graph, seed_node)
            new_node = generate_new_node(seed_node, inspiration_node, model, graph)
            if new_node:
                graph[new_node.id] = new_node

        save_graph(graph, graph_output_path)


def visualize_graph(input_path, output_path):
    # Load the graph from input_path
    with open(input_path, 'r') as f:
        graph_data = json.load(f)

    # Build node dictionary and edge list
    nodes = {}
    edges = []
    for node_data in graph_data['nodes']:
        node_id = node_data['id']
        node_type = node_data['type']
        content = node_data['content']
        parents = node_data.get('parents', [])
        nodes[node_id] = {
            'type': node_type,
            'content': content,
            'parents': parents
        }
        # Add edges from parents to this node
        for parent_id in parents:
            edges.append((parent_id, node_id))

    G = nx.DiGraph()
    for node_id, data in nodes.items():
        G.add_node(node_id, **data)
    G.add_edges_from(edges)

    # Initialize distances
    for node_id in G.nodes():
        G.nodes[node_id]['distance'] = None

    idea_nodes = [node_id for node_id, data in G.nodes(data=True) if data['type'] == 'idea']
    max_distance = 0

    # BFS to compute distances from original idea nodes
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
        label = node_id
        node_attrs = {'label': label}
        node_type = data['type']

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
            dist = data['distance']
            t = dist / max(1, max_distance)
            color_rgb = interpolate_color(light_blue_rgb, dark_blue_rgb, t)
            color_hex = rgb_to_hex(color_rgb)
            node_attrs['fillcolor'] = color_hex
            node_attrs['shape'] = 'box'
            node_attrs['fontsize'] = '16'
            # Dark text on light background, light text on dark background
            node_attrs['fontcolor'] = rgb_to_hex(interpolate_color((100, 100, 100), (255, 255, 255), t))
            # Compute node size based on number of lines
            content = data['content']
            num_lines = len(content.strip().split('\n'))
            base_size = 0.5
            size = base_size + num_lines**0.6 * 0.2
            #max_size = 4.0
            #size = min(size, max_size)
            node_attrs['width'] = str(size)
            node_attrs['height'] = str(size)
            node_attrs['fixedsize'] = 'true'

        else:
            node_attrs['shape'] = 'circle'

        dot.node(node_id, **node_attrs)

    for u, v in G.edges():
        dot.edge(u, v)

    dot.save(output_path)
    print(f"Graphviz visualization saved to {output_path}")


def trace_node(graph_path, input_node_id):
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)

    nodes = {}
    for node_data in graph_data['nodes']:
        node_id = node_data['id']
        nodes[node_id] = node_data

    if input_node_id not in nodes:
        print(f"No node '{input_node_id}'")
        return

    def find_path_to_idea(node_id, path):
        node = nodes[node_id]
        path.append(node)
        parents = node.get('parents', [])
        if not parents:
            if node['type'] == 'idea':
                return
        else:
            find_path_to_idea(parents[0], path)

    path = []
    find_path_to_idea(input_node_id, path)

    for node in reversed(path):
        node_id = node['id']
        node_type = node['type']
        content = node['content']
        print(f"Node ID: {node_id}, Type: {node_type}")
        print(content)
        print('---')


def main():
    parser = argparse.ArgumentParser(description="Synthesize annotated Dafny programs based on a seed set.")
    parser.add_argument("--model", type=str, help="LLM to use", default='gpt-4o-2024-08-06')
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations (K).")
    parser.add_argument("--samples", type=int, default=5, help="Number of seed nodes to sample in each iteration (M).")
    parser.add_argument("--graph-input", type=str, help="Path to existing graph JSON file.")
    parser.add_argument("--ideas-input", type=str, help="Path to initial ideas file.")
    parser.add_argument("--graph-output", type=str, help="Path to output graph JSON file.", default='graph.json')
    parser.add_argument("--plot-graph", action='store_true', help="Generate a GraphViz visualization of the graph.")
    parser.add_argument("--trace-node", type=str, help="Show the path from an idea to the given node.")

    args = parser.parse_args()

    if args.graph_input:
        graph = load_graph(args.graph_input)
    elif args.ideas_input:
        graph = initialize_graph_from_ideas(args.ideas_input)
    else:
        print("Error: Must provide either --graph-input or --ideas-input")
        return

    if args.plot_graph:
        visualize_graph(args.graph_input, 'graph.dot')
    elif args.trace_node:
        trace_node(args.graph_input, args.trace_node)
    else:
        run_iterations(graph, args.iterations, args.samples, args.model, args.graph_output)
        print(f"Graph saved to {args.graph_output}")


if __name__ == "__main__":
    main()

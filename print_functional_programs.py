import sys
from edit_graph import EditGraph
from editor import _extract_axiom

programs_file = sys.argv[1]

graph = EditGraph.load_graph(programs_file)

full_programs = []
for node in graph.nodes.values():
    if node.type == 'program':
        print(node.content)
        print('-' * 100)
        if _extract_axiom('function ', node.content) is None and _extract_axiom('lemma {:axiom} ', node.content) is None:
            full_programs.append(node.content)
print('\nFound', len(full_programs), 'full programs')
for p in full_programs:
    print(p)
    print('-' * 100)
print('\nFound', len(full_programs), 'full programs')
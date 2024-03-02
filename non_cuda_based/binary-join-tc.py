edges = [
    {'source': 1, 'target': 2},
    {'source': 2, 'target': 3},
    {'source': 3, 'target': 4}
]

def binary_join_transitive_closure(edges):
    connections = set((edge['source'], edge['target']) for edge in edges)
    while True:
        new_connections_found = False
        for (source, target) in list(connections):
            for edge in edges:
                # Check if we can extend an existing connection.
                if edge['source'] == target:
                    new_connection = (source, edge['target'])
                    if new_connection not in connections:
                        connections.add(new_connection)
                        new_connections_found = True
        if not new_connections_found:
            break
    return connections

transitive_closure_binary = binary_join_transitive_closure(edges)

print(f"Transitive closure: {transitive_closure_binary}")

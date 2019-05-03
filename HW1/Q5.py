
import networkx as nx


def main():
    # Create the friendship graph using network package
    graph = nx.Graph()
    with open('./data/karate_club_graph.txt', 'r') as graphData:
        for entry in graphData:
            if entry.startswith('#'):
                continue
            entry_data = entry.rstrip().split(' ')
            node1 = int(entry_data[0])
            node2 = int(entry_data[1])
            graph.add_edge(node1, node2, capacity=1.0)

    partition = nx.minimum_cut(graph, 1, 34)
    partition_sets = partition[1]
    i = 1
    for club in partition_sets:
        print(f'Club {i} Members: {club}')
        i += 1


if __name__ == '__main__': main()

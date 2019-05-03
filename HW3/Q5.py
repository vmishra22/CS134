import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from collections import deque
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
import sys
import math
from heapq import *

# All networks in this question are unweighted.

# Q5a (4 points)
'''
Write a function \texttt{load\_network(filename)} (type signature: \texttt{str $\to$ network}) that accepts a single string parameter
(a path/name to a file
containing an encoded network) and returns a Python representation of that network (you may choose what this representation
should be: we don't care whether you use an adjacency list or matrix, for example, or whether you use dictionaries or lists, etc).
Assume that the file encodes the network with numbered nodes, and contains a single edge per line in the format
`initial node number'-`whitespace'-`terminal node number'. Also assume that all network links are undirected: if you encounter a link
0 to 1, be sure to account for the return link 1 to 0 even if that link is not explicitly listed in the file.

For example, suppose you had a network file called \texttt{sample.txt} in your
\texttt{datasets} directory, with the following contents:

0 1
0 2
1 2
2 0

Then, invoking \texttt{load\_network(`datasets/sample.txt')} might return \texttt{\{ 0 : [1,2], 1: [0,2], 2: [0,1] \}}.
'''


def load_network(filename):
    G = {}
    with open(f'./datasets/{filename}', 'r') as graphData:
        for entry in graphData:
            adjList = []
            if entry.startswith('#'):
                continue
            entry_data = entry.rstrip().split(' ')
            node1 = int(entry_data[0])
            node2 = int(entry_data[1])
            if node1 not in G:
                G[node1] = []
                G[node1].append(node2)
            else:
                if node2 not in G[node1]:
                    G[node1].append(node2)
            if node2 not in G:
                G[node2] = []
                G[node2].append(node1)
            else:
                if node1 not in G[node2]:
                    G[node2].append(node1)
    return G


# Q5b (4 points)
'''
Write a function \texttt{diameter(network)} (type signature: \texttt{network $\to$ int}) that accepts
a network (in your chosen Python representation), and reports its diameter. For example,
invoking \texttt{diameter(load\_network(`datasets/sample.txt'))} should return \texttt{1}. (Hint: you may
want to first write a helper function to calculate the shortest-path distances between all pairs of nodes.)
'''


def all_pairs_shortest_distance(graph):
    dist = {}
    for s in graph:
        dist[s] = {}
        for t in graph:
            dist[s][t] = math.inf

        dist[s][s] = 0
        for t in graph[s]:
            dist[s][t] = 1

    for k in graph:
        for s in graph:
            for t in graph:
                if dist[s][k] + dist[k][t] < dist[s][t]:
                    dist[s][t] = dist[s][k] + dist[k][t]
    return dist


def diameter(network):
    distances = all_pairs_shortest_distance(network)
    graph_diameter = 0
    for i in distances:
        for j in distances[i]:
            if distances[i][j] > graph_diameter:
                graph_diameter = distances[i][j]
    return graph_diameter


# Q5c (4 point)
'''
Write a function \texttt{avg\_distance(network)} (type signature: \texttt{network $\to$ float}) that accepts
a network (in your chosen Python representation), and reports the average distance between pairs of nodes. For example,
invoking \texttt{avg\_distance(load\_network(`datasets/sample.txt'))} should return \texttt{1.0}. (Hint: you may
want to use the same all-pairs shortest-paths helper from above.)
'''


def avg_distance(network):
    distances = all_pairs_shortest_distance(network)
    nodes_count = len(network)
    sum_distances = 0
    num_pairs = nodes_count * (nodes_count - 1) / 2
    for i in distances:
        for j in distances[i]:
            if i == j:
                continue
            edge_length = distances[i][j]
            sum_distances += edge_length
            # num_pairs += 1
            j_map = distances[j]
            del j_map[i]

    return sum_distances / num_pairs


# Q5d (4 points)
'''
For each of the network files \texttt{sparshblock.txt} (a friendship network of Sparsh's blocking group before they came to Harvard),
\texttt{karateclub.txt} (from your previous problem set), \texttt{dolphins.txt} (a dolphin pod tracking network), \texttt{lesmis.txt} (a character map
from Les Miserables), and \texttt{airports.txt} (a network of connecting flights among US airports in 1992), compute the diameter and the average distance,
and plot these on a graph. What do you notice?
'''


def plot_dataset_graphs():
    xValues = list()
    yValues = list()
    listFiles = ['sparshblock.txt', 'karateclub.txt', 'dolphins.txt', 'lesmis.txt', 'airports.txt']
    networks = []
    networks = list(map(lambda x: load_network(x), listFiles))
    xValues = list(map(lambda x: avg_distance(x), networks))
    yValues = list(map(lambda x: diameter(x), networks))

    plt.xlabel('Average Distances')
    plt.ylabel('Diameter')
    plt.title(f'Graph average distance vs. diameter')
    plt.scatter(xValues, yValues)
    plt.show()
    pass


# Q5e (3 points)
'''
Write a function \texttt{distance(network, u, v)} (type signature: \texttt{network $\times$ int $\times$ int $\to$ int}) that accepts
a network (in your chosen Python representation), an initial node $u$, and a terminal node $v$, and reports the
shortest-path distance in that network from $u$ to $v$. For example,
invoking \texttt{distance(load\_network(`datasets/sample.txt'), 0, 1)} should return \texttt{1}.
'''


def distance(network, u, v):
    dist = {}
    heapq = []
    heappush(heapq, (0, u))
    for s in network:
        dist[s] = math.inf
    dist[u] = 0
    visited = set()
    while heapq:
        (distance_val, a1) = heappop(heapq)
        if a1 not in visited:
            visited.add(a1)

            if a1 == v:
                return distance_val
            for t in network[a1]:
                if t in visited:
                    continue
                if dist[t] > dist[a1] + 1:
                    dist[t] = dist[a1] + 1
                    heappush(heapq, (dist[t], t))

    return dist[v]


# Q5f (3 points)
'''
In this question, we're still interested in average distances in a network, but these networks are too big to
conveniently calculate average distance by brute force. Therefore, we must use what we learned about sampling
and error bounds in class to estimate the true values. Please estimate the true value of average distance
for the following two networks: \texttt{enron.txt} and \texttt{epinions.txt}. Please submit your results, as well as your
code and a short written explanation of how you approached the problem.
'''


def empirical_average_distance():
    listFiles = ['enron.txt', 'epinions.txt']
    networks = list(map(lambda x: load_network(x), listFiles))
    trials = 100
    average_distance = []
    sum_distances = 0
    for g in networks:
        g_size = len(g.keys())
        succ_trials = 0
        for i in range(trials):
            u = int(np.random.randint(0, g_size, 1))
            while g.get(int(u)) is None:
                u = int(np.random.randint(0, g_size, 1))

            v = int(np.random.randint(0, g_size, 1))
            while g.get(int(v)) is None:
                v = int(np.random.randint(0, g_size, 1))

            dist_val = distance(g, int(u), int(v))
            if dist_val is not math.inf:
                sum_distances += dist_val
                succ_trials += 1

        average_distance.append(sum_distances / succ_trials)

    print(average_distance)
    pass


# Public Unit Tests (uncomment as you go)


assert (diameter(load_network('sample_public.txt')) == 5)
assert (abs(avg_distance(load_network('sample_public.txt')) - 2.333) < 0.1 or abs(
    avg_distance(load_network('sample_public.txt')) - 2.074) < 0.1)
assert (distance(load_network('sample_public.txt'), 0, 8) == 3)


def main():
    plot_dataset_graphs()
    empirical_average_distance()


if __name__ == '__main__': main()

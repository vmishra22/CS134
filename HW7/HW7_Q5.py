import operator
import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from collections import deque, Counter
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
import sys
import math
from heapq import *


def load_network(filename):
    G = {}
    GNodeSet = set()
    with open(f'./datasets/{filename}', 'r') as graphData:
        for entry in graphData:
            adjList = []
            if entry.startswith('#'):
                continue
            entry_data = entry.rstrip().split()
            node1 = int(entry_data[0])
            node2 = int(entry_data[1])
            prob = float(entry_data[2])

            if node1 not in G:
                G[node1] = []
                G[node1].append((node2, prob))
                GNodeSet.update([node1, node2])
            else:
                value = [v for i, v in enumerate(G[node1]) if v[0] == node2]
                if len(value) == 0:
                    G[node1].append((node2, prob))
                    GNodeSet.update([node2])
    return G, len(GNodeSet)


def genRandGraph(weights):
    sampled_graph = {}
    numEdges = 0
    p_list = []
    set_realized_node = set()
    for node in weights:
        list_edges = weights[node]
        numEdges += len(list_edges)
        for e in list_edges:
            p_list.append(e[1])

    rand_list = np.random.uniform(0.0, 1.0, numEdges)
    comp = np.less_equal(rand_list, np.asarray(p_list))

    cur_index = 0
    for node in weights:
        list_edges = weights[node]
        nEdges = len(list_edges)
        eRealized_list = list(comp[cur_index:cur_index + nEdges])
        realized_edges = [list_edges[ind] for ind, x in enumerate(eRealized_list) if
                          eRealized_list[ind]]

        cur_index += nEdges
        if len(realized_edges) > 0:
            sampled_graph[node] = realized_edges
            set_realized_node.add(node)
            realized_nodes = [x[0] for x in realized_edges]
            set_realized_node.update(realized_nodes)

    return sampled_graph, len(set_realized_node)


def sampleInfluence(G, S, m):
    r = [0] * m
    for i in range(m):
        realized_graph, setRSize = genRandGraph(G)

        GKeys = realized_graph.keys()
        reachable = {key: False for key in GKeys}
        for sNode in S:
            reachable[sNode] = True

        for S_node in S:
            queue = deque()
            queue.append(S_node)
            while len(queue) != 0:
                current = queue.popleft()
                if current in realized_graph:
                    for graphNode in realized_graph[current]:
                        node = graphNode[0]
                        if node in reachable:
                            if not reachable[node]:
                                reachable[node] = True
                                queue.append(node)
                        else:
                            reachable[node] = True
                            queue.append(node)

        reachable_list = [p for p in reachable if reachable[p]]
        r[i] = len(reachable_list)

    total_count = sum(r)
    return total_count / m


def find_S(G, sampleCount):
    solution_set = set()
    while len(solution_set) < 5:
        marginal_value_node = {}
        for node in G:
            if node not in solution_set:
                current_set = solution_set
                solution_set.add(node)
                value = sampleInfluence(G, solution_set, sampleCount) - sampleInfluence(G, current_set, sampleCount)
                solution_set.remove(node)
                marginal_value_node[node] = value
                print(f'f_S(a): {value}, S = {solution_set}, a = {node}')

        max_value_node = max(marginal_value_node.items(), key=operator.itemgetter(1))[0]
        solution_set.add(max_value_node)

    return solution_set


def main():
    weighted_graph, nGraphNodes = load_network("network.txt")
    value = [v for i, v in enumerate(weighted_graph[42]) if v[0] == 75]
    print(f'A. probability for edge 42 -> 75: {value[0][1]}')

    i = 0
    numSampledGraphs = 100
    numNodes = 0
    while i < numSampledGraphs:
        realized_graph, setRSize = genRandGraph(weighted_graph)
        numNodes += setRSize
        i += 1
    print(f'B. Average number of nodes in the realized graph: {numNodes/numSampledGraphs}')

    S_sample = [17, 23, 42, 2017]
    m = 500
    f_S = sampleInfluence(weighted_graph, S_sample, m)
    print(f'C. f(S) from sampleInfluence: {f_S}')

    # Using number of samples calculated from problem 2(d)
    k = 5
    eps = 90
    a = pow(nGraphNodes, 2.)
    b = pow(k, 2.)
    c = math.log(k * pow(nGraphNodes, 2.))
    d = 2. * pow(eps, 4.)
    sampleCount = (a * b * c) / d

    # For the above values sample count is 67.
    S = find_S(weighted_graph, int(sampleCount))
    print(f'D. Solution set S: {S}')


if __name__ == '__main__':
    main()

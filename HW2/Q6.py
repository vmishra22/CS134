import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from collections import deque
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path


def create_random_graph(n, p):
    G = []
    skipDict = {}
    for i in range(n):
        adjList = []
        skipDict[i] = []
        currentVerticesLen = len(G)
        for m in range(n):
            if i == m:
                continue
            else:
                if m < currentVerticesLen and i in G[m]:
                    adjList.append(m)
                else:
                    if skipDict.get(m) is None or (skipDict.get(m) is not None and i not in skipDict[m]):
                        randVal = np.random.uniform(0.0, 1.0, 1)
                        if randVal <= p:
                            adjList.append(m)
                        else:
                            skipDict[i].append(m)
        G.append(adjList)
    return G


def search(G, visited, vertIndex):
    visited[vertIndex] = 1
    neighbors = G[vertIndex]
    for nIndex in neighbors:
        if visited[nIndex] == 0:
            search(G, visited, nIndex)


def find_components(G):
    numConnectedComponents = 0
    n = len(G)
    visited = [0 for i in range(n)]
    for vertIndex in range(n):
        if visited[vertIndex] == 0:
            numConnectedComponents += 1
            search(G, visited, vertIndex)

    return numConnectedComponents


def simulate_connectivity(n, p, nTrials):
    probabilityMap = {}
    for pi in range(nTrials):
        nTotalComps = 0
        nConnectedGraphs = 0
        for i in range(nTrials):
            G = create_random_graph(n, p)
            nComps = find_components(G)
            nTotalComps += nComps
            if nComps == 1:
                nConnectedGraphs += 1
        print(f'Parameter p: [{p}]')
        averageConnComps = nTotalComps/nTrials
        empProbability = nConnectedGraphs/nTrials
        print(f'Average number of CC: {averageConnComps}')
        print(f'Connectivity: {empProbability}')
        probabilityMap[pi] = [averageConnComps, empProbability]
        p += 1.0e-3
    return probabilityMap


def main():

    Graph = create_random_graph(50, 0.05)
    numComponents = find_components(Graph)
    print(f'Number of connected components for n=50, p=0.5: {numComponents}')

    n = 100
    p = 0.001
    nTrials = 100
    probabilityMap = simulate_connectivity(n, p, nTrials)
    x_values = [0.001+i*0.001 for i in range(nTrials)]
    y1_values = [probabilityMap[i][0] for i in range(nTrials)]
    y2_values = [probabilityMap[i][1] for i in range(nTrials)]

    plt.ylabel('Average connected components')
    plt.xlabel('probability')
    plt.title(f'Average # of connected components vs Probability')
    plt.plot(x_values, y1_values)
    plt.show()

    plt.ylabel('Empirical probability')
    plt.xlabel('probability')
    plt.title(f'Empirical probability vs Probability')
    plt.plot(x_values, y2_values)
    plt.show()

if __name__ == '__main__': main()

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
from scipy.sparse import dia_matrix
import operator


def load_network(filename):
    G = {}
    with open(f'./{filename}', 'r') as graphData:
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

        nNodes = len(G.keys())
        lSize = 0
        for n in G:
            lSize += len(G[n])
        avg_out_degree = lSize / nNodes

        print(f'numNodes: {nNodes}, avg_out_degree: {avg_out_degree}')

    return G


def pageRankIter(g, d):
    N = len(g.keys())
    j = 0
    nIMap = {}
    iNMap = {}
    d_new = {}

    for i in g:
        nIMap[i] = j
        iNMap[j] = i
        j += 1

    indicesMatrix = {}
    for i in g:
        lneighbors = g[i]
        neighborsIndex = [nIMap[xn] for xn in lneighbors]
        indicesMatrix[nIMap[i]] = neighborsIndex

    scoreColumn = {}
    for dn in d.keys():
        scoreColumn[nIMap[dn]] = d[dn]

    PS = np.zeros(N, dtype=np.float16)

    for iM in indicesMatrix.keys():
        neighborsIndexList = indicesMatrix[iM]
        nNeighbors = len(neighborsIndexList)
        if nNeighbors == 0:
            PS[iM] = 1.0 * scoreColumn[iM]
        else:
            for nM in neighborsIndexList:
                value = float(1. / nNeighbors) * scoreColumn[nM]
                PS[iM] += value

    for iP in range(PS.shape[0]):
        d_new[iNMap[iP]] = PS[iP]

    return d_new


def basicPR(g, d, k):
    d_new = d
    for i in range(k):
        print(f'k: {k}, i: {i}')
        d_new = pageRankIter(g, d_new)

    return d_new


def iterative_basicPR(g, d):
    k_values = [10, 50, 200]
    for k in k_values:
        d_new_k = basicPR(g, d, k)
        PS_iter_k = d_new_k.values()
        trunc_PS_iter_k = [i for i in PS_iter_k if i >= 0.0000020 and i <= 0.0000035]
        plt.hist(trunc_PS_iter_k, bins=15)
        plt.xlabel('Page scores')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Page scores after k: {k} iterations')
        plt.show()


def apply_scalar(x, s, size):
    return x * s + (1 - s) / size


def pageRankScaledIter(g, d, s):
    N = len(g.keys())
    j = 0
    nIMap = {}
    iNMap = {}
    d_new = {}

    for i in g:
        nIMap[i] = j
        iNMap[j] = i
        j += 1

    indicesMatrix = {}
    for i in g:
        lneighbors = g[i]
        neighborsIndex = [nIMap[xn] for xn in lneighbors]
        indicesMatrix[nIMap[i]] = neighborsIndex

    scoreColumn = {}
    for dn in d.keys():
        scoreColumn[nIMap[dn]] = d[dn]

    PS = np.zeros(N, dtype=np.float16)

    for iM in indicesMatrix.keys():
        neighborsIndexList = indicesMatrix[iM]
        nNeighbors = len(neighborsIndexList)
        if nNeighbors == 0:
            PS[iM] = ((1.0 * s) + (1 - s) / N) * scoreColumn[nM]
        else:
            for nM in neighborsIndexList:
                value = ((float(1. / nNeighbors) * s) + (1 - s) / N) * scoreColumn[nM]
                PS[iM] += value

    for iP in range(PS.shape[0]):
        d_new[iNMap[iP]] = PS[iP]

    return d_new


def scaledPR(g, d, k, s):
    d_new = d
    for i in range(k):
        print(f'k: {k}, i: {i}')
        d_new = pageRankScaledIter(g, d_new, s)

    return d_new


def iterative_scaledPR(g, d):
    k_values = [10, 50, 200]
    s = 0.85
    for k in k_values:
        d_new_k = scaledPR(g, d, k, s)
        PS_iter_k = d_new_k.values()
        trunc_PS_iter_k = [i for i in PS_iter_k if i >= 0.00001]
        plt.hist(PS_iter_k, bins=15)
        plt.xlabel('Page scores')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Scaled Page scores after k: {k} iterations')
        plt.show()


def load_link(filename, s):
    node_list = []
    with open(f'./{filename}', 'r') as linkData:
        for entry in linkData:
            adjList = []
            if entry.startswith('#'):
                continue
            entry_data = entry.rstrip().split(' ')
            node1 = int(entry_data[0])
            node2 = entry_data[1]

            if s in node2:
                node_list.append(node1)

    return node_list


def main():
    fileName = 'google.txt'
    G = load_network(fileName)
    nNodes = len(G.keys())

    d_start = {n: float(1 / nNodes) for n in G}
    d_new_1 = pageRankIter(G, d_start)
    PS_iter_1 = d_new_1.values()
    trunc_PS_iter_1 = [i for i in PS_iter_1 if i >= 0.0000020 and i <= 0.000003]
    plt.hist(trunc_PS_iter_1, bins=15)
    plt.xlabel('Page scores')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Page scores after One iteration')
    plt.show()

    iterative_basicPR(G, d_start)
    iterative_scaledPR(G, d_start)

    s = '34'
    matching_nodes = load_link('links.txt', s)
    d_search_node_PS = scaledPR(G, d_start, 100, 0.85)
    matching_nodes_with_value = {k: v for k, v in d_search_node_PS.items() if k in matching_nodes}
    sorted_matching_nodes_with_value = sorted(matching_nodes_with_value.items(), key=operator.itemgetter(1),
                                              reverse=True)
    print(f'{sorted_matching_nodes_with_value[0:5]}')


if __name__ == '__main__':
    main()

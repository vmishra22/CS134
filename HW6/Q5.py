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


def load_graph(filename):
    G = []
    prob_list = []
    with open(f'./datasets/{filename}', 'r') as graphData:
        for entry in graphData:
            if entry.startswith('#'):
                continue
            entry_data = entry.rstrip().split()
            node1 = int(entry_data[0])
            node2 = int(entry_data[1])
            prob = float(entry_data[2])
            current_len = len(G)

            if node1 == current_len:
                G.insert(node1, [])
                prob_list.insert(node1, [])
            elif node1 > current_len:
                i = current_len
                while i <= node1:
                    G.insert(i, [])
                    prob_list.insert(i, [])
                    i += 1
            if node2 not in G[node1]:
                G[node1].append(node2)
                prob_list[node1].append(prob)

            current_len = len(G)
            if node2 == current_len:
                G.insert(node2, [])
                prob_list.insert(node2, [])
            elif node2 > current_len:
                i = current_len
                while i <= node2:
                    G.insert(i, [])
                    prob_list.insert(i, [])
                    i += 1
            if node1 not in G[node2]:
                G[node2].append(node1)
                prob_list[node2].append(prob)

    return G, prob_list


def load_cascades(filename):
    C = []
    timeStepsList = []
    listOfTimeStepsList = []
    rootNext = False
    with open(f'./datasets/{filename}', 'r') as cascadeData:
        for entry in cascadeData:
            if "New Cascade" in entry:
                if len(listOfTimeStepsList) > 0:
                    C[lenC - 1].append(listOfTimeStepsList)

                C.append([])
                rootNext = True
                listOfTimeStepsList = []
                continue
            elif "Time Step" in entry:
                listOfTimeStepsList.append([])
                continue
            elif "EOF" in entry:
                if len(listOfTimeStepsList) > 0:
                    C[lenC - 1].append(listOfTimeStepsList)
                break

            entry_data = entry.rstrip().split()
            entry1 = int(entry_data[0])
            if rootNext:
                lenC = len(C)
                C[lenC - 1].append(entry1)
                rootNext = False
            else:
                entry2 = int(entry_data[1])
                numTimeStepsList = len(listOfTimeStepsList)
                listOfTimeStepsList[numTimeStepsList - 1].append([entry1, entry2])
    return C


def estimate_probabilities(graph, cascade):
    empirical_prob_list = []
    num_edge_activation = {}
    num_edge_access = {}

    i = 0
    for node in graph:
        if i not in num_edge_access and len(node) > 0:
            num_edge_access[i] = [0] * len(node)
        i += 1
    # Each cIndex belongs to a new Cascade( a new root)
    for cIndex in cascade:
        root = cIndex[0]
        listOfTimeStepsList = cIndex[1]
        for timeStepList in listOfTimeStepsList:
            node1 = -1
            prev_node1 = -1
            for edge in timeStepList:
                node1 = edge[0]
                node2 = edge[1]
                neighborList = graph[node1]

                if node1 not in num_edge_activation:
                    num_edge_activation[node1] = [0] * len(neighborList)

                if prev_node1 != node1:
                    incr_list = [x + 1 for x in num_edge_access[node1]]
                    num_edge_access[node1] = incr_list

                if node2 in neighborList:
                    activated_node_index = neighborList.index(node2)
                    num_edge_activation[node1][activated_node_index] += 1

                prev_node1 = node1

    for i in num_edge_access:
        current_len = len(empirical_prob_list)
        if i == current_len:
            empirical_prob_list.insert(i, [])
        elif i > current_len:
            j = current_len
            while j <= i:
                empirical_prob_list.insert(j, [])
                j += 1

        result = []
        end_index = len(num_edge_access[i])
        for k in range(end_index):
            if num_edge_access[i][k] == 0:
                result.append(-1)
                continue
            result.append(num_edge_activation[i][k] / num_edge_access[i][k])

        empirical_prob_list[i] = result

    return empirical_prob_list


def avg_error(graph, cascade, real_prob_list):
    emp_prob_list = estimate_probabilities(graph, cascade)
    end_index = len(real_prob_list)
    error_sum = 0.
    num_edges = 0
    for k in range(end_index):
        pList = real_prob_list[k]
        if len(pList) > 0:
            end_index1 = len(pList)
            for k1 in range(end_index1):
                if emp_prob_list[k][k1] == -1:
                    error_sum += 1
                else:
                    error_sum += abs(emp_prob_list[k][k1] - pList[k1])
                num_edges += 1
    return error_sum / num_edges


def main():

    list_graphs = ['graph1.txt', 'graph2.txt']
    list_cascades = ['cascades1.txt', 'cascades2.txt']
    graph_prob = list(list(map(lambda x: load_graph(x), list_graphs)))
    cascades = list(map(lambda x: load_cascades(x), list_cascades))

    i = 0
    while i < len(list_graphs):
        num_cascades = len(cascades[i])
        list_avg_learning_error = []
        k = 1
        while k <= num_cascades:
            avg_learning_error = avg_error(graph_prob[i][0], cascades[i][0:k], graph_prob[i][1])
            list_avg_learning_error.append(avg_learning_error)
            k += 1

        x_values = range(1, num_cascades + 1)
        plt.figure(i + 1)
        plt.ylabel('Average error')
        plt.xlabel('Number of cascades')
        plt.title(f'Average error vs Number of cascades: {num_cascades}')
        plt.plot(x_values, list_avg_learning_error)
        plt.show()
        i += 1


if __name__ == '__main__':
    main()

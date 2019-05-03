# Harvard SEAS CS134: Network Science
# 2018 Fall
# Profs Michael Mitzenmacher and Yaron Singer
# TF David Miller
# Problem Set 4: Small-World Networks
# Question 4. (50 points) Coding: Comparing Power Law Exponents and Friendship Paradox Ratios in Real-World Networks

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

# N.B. Remember that graphs/networks mean the same thing, vertices/nodes mean the same thing, edges/links mean the same thing.
# See the problem set for a few more notes / clarifications

# Q5a (10 points)
'''
Write two functions, \texttt{findNodeDegreesDirected(filename)} and \texttt{findNodeDegreesUndirected(filename)}
(type signature: \texttt{str $\to$ dict}),  that each accept a single string parameter (a path/name of a file containing 
an encoded network) and return a dictionary where each key corresponds to a node in the graph and each value is the degree 
of that node (in-degree if the graph is directed). Assume that the file encodes the network with numbered nodes, and that 
there will be a commented line at the beginning of each file indicating whether the graph is directed or undirected.

For example, for a directed network file called \texttt{blah.txt} with the following contents:

\texttt{0 1\\
0 2\\
1 2\\
2 0
}

invoking \texttt{findNodeDegreesDirected(`blah.txt')} might return $\{ 0 : 1, 1: 1, 2: 2 \}$.
'''


def findNodeDegreesDirected(filename):
    dict_node_degree = {}
    directedGraph = False
    with open(f'./datasets/{filename}', 'r') as graphData:
        for entry in graphData:
            if entry.startswith('#'):
                handle = entry.split()
                if handle[1] == 'Undirected':
                    break
                else:
                    continue
            entry_data = entry.rstrip().split()
            node1 = int(entry_data[0])
            node2 = int(entry_data[1])

            if node2 not in dict_node_degree:
                dict_node_degree[node2] = 1
            else:
                dict_node_degree[node2] += 1

    return dict_node_degree


def findNodeDegreesUndirected(filename):
    dict_node_degree = {}
    with open(f'./datasets/{filename}', 'r') as graphData:
        for entry in graphData:
            if entry.startswith('#'):
                handle = entry.split()
                if handle[1] == 'Directed':
                    break
                else:
                    continue
            entry_data = entry.rstrip().split()
            node1 = int(entry_data[0])
            node2 = int(entry_data[1])

            if node2 not in dict_node_degree:
                dict_node_degree[node2] = 1
            else:
                dict_node_degree[node2] += 1
            if node1 not in dict_node_degree:
                dict_node_degree[node1] = 1
            else:
                dict_node_degree[node1] += 1

    return dict_node_degree


# Q5b (8 points)
'''
Write a function \texttt{findExponentOLS(graph)} (type signature: \texttt{dict $\to$ float}) that accepts
a dictionary as described in part a, and returns an estimate $\hat{\alpha}$ for the exponent of a power law 
curve fitted to the degree distribution of the network described by the graph. The estimate should be calculated 
through ordinary least squares regression (see Section 4 notes for details). For example, the function call 
\texttt{findExponentOLS(findNodeDegreesDirected(`sample.txt'))} should return approximately \texttt{0.908}.
'''


def findExponentOLS(graph):
    nDegrees = Counter(graph.values())
    node_degrees = np.array(list(nDegrees.keys()))
    log_x_values = np.log(node_degrees)
    array_of_ones = np.array([1] * len(nDegrees))
    x_vals = np.column_stack((log_x_values, array_of_ones))
    y_vals = np.array(list(nDegrees.values()))
    log_y_values = np.log(y_vals)
    x_T_x = np.dot(np.transpose(x_vals), x_vals)
    w1 = np.dot(np.linalg.inv(x_T_x), np.dot(np.transpose(x_vals), log_y_values))[0]
    return w1


# Q5c (8 points)
'''
Write a function \texttt{findExponentMML(graph)} (type signature: \texttt{dict $\to$ float}) that accepts
a dictionary as described in part a, and returns an estimate $\hat{\alpha}$ for the exponent of a power law 
curve fitted to the degree distribution of the network described by the graph. The estimate should be calculated 
through the method of maximum likelihood (see Section 4 notes for details). For example, the function call 
\texttt{findExponentMML(findNodeDegreesDirected(`sample.txt'))} should return approximately \texttt{3.57}. 
(Note: For this part, you should assume an integral power law distribution.) Which method do you think provides a 
better estimate for $\alpha$? Why?
'''


def findExponentMML(graph):
    d_min = 1.
    nDegrees = Counter(graph.values())
    num_degrees = len(nDegrees)
    node_degrees = np.array(list(nDegrees.keys()))
    log_x_values = np.log(node_degrees)
    list_val = [i for i in log_x_values if i != 0.0]
    log_values = np.log(node_degrees / (d_min - 0.5))
    sum_values = np.sum(log_values)
    alpha_hat = 1 + num_degrees * (1. / sum_values)
    return alpha_hat


# Q5d (10 points)
'''
Write two functions, \texttt{directedFriendshipParadox(filename)} and \\ \texttt{undirectedFriendshipParadox(filename)}
(type signature: \texttt{str $\to$ float}),  that each accept a single string parameter (a path/name of a file containing
an encoded network) and return the ``Friendship Paradox Ratio'' of the network, defined as:
$$\frac{\text{average degree of a node in the network}}{\text{average degree of a node's neighbors in the network (averaged over all nodes)}}$$
For directed graphs, you should calculate average in-degree instead of average degree. For example, the function call 
\texttt{directedFriendshipParadox(`sample.txt')} should return approximately \texttt{0.87}. Note: for larger 
networks (e.g. greater than 1000 nodes), you should randomly sample at least 500 nodes with replacement for your calculations. 
You may assume that such larger network files will list their edges in numerical order (i.e. all of the edges coming from node 1, 
followed by those coming from node 2, etc.).
'''


def directedFriendshipParadox(filename):
    node_neighbors_map = {}
    with open(f'./datasets/{filename}', 'r') as graphData:
        for entry in graphData:
            if entry.startswith('#'):
                handle = entry.split()
                if handle[1] == 'Undirected':
                    break
                else:
                    continue
            entry_data = entry.rstrip().split()
            node1 = int(entry_data[0])
            node2 = int(entry_data[1])
            if node1 not in node_neighbors_map:
                node_neighbors_map[node1] = []
            if node2 not in node_neighbors_map:
                node_neighbors_map[node2] = []
            # since in-degree neighbors are to be considered ..
            node_neighbors_map[node2].append(node1)
    num_graph_nodes = len(node_neighbors_map)
    sum_node_degree = 0
    average_degree_neighbor = 0.
    if num_graph_nodes > 1000:
        # sample 500 nodes of the graph
        node_vals = np.array(list(node_neighbors_map.keys()))
        random_arr = np.random.choice(node_vals, 500)

        for rand_node in random_arr:
            node_neighbors = node_neighbors_map[rand_node]
            sum_node_degree += len(node_neighbors)
            if len(node_neighbors):
                average_degree_neighbor += (sum([len(node_neighbors_map[n]) for n in node_neighbors]) / len(node_neighbors))
    else:
        for i in node_neighbors_map:
            node_neighbors = node_neighbors_map[i]
            sum_node_degree += len(node_neighbors)
            if len(node_neighbors):
                average_degree_neighbor += (sum([len(node_neighbors_map[n]) for n in node_neighbors]) / len(node_neighbors))

    ratio = sum_node_degree / average_degree_neighbor
    return ratio

def undirectedFriendshipParadox(filename):
    node_neighbors_map = {}
    with open(f'./datasets/{filename}', 'r') as graphData:
        for entry in graphData:
            if entry.startswith('#'):
                handle = entry.split()
                if handle[1] == 'directed':
                    break
                else:
                    continue
            entry_data = entry.rstrip().split()
            node1 = int(entry_data[0])
            node2 = int(entry_data[1])
            if node1 not in node_neighbors_map:
                node_neighbors_map[node1] = []
            if node2 not in node_neighbors_map:
                node_neighbors_map[node2] = []
            if node1 != node2:
                node_neighbors_map[node2].append(node1)
                node_neighbors_map[node1].append(node2)
            else:
                node_neighbors_map[node1].append(node2)

    num_graph_nodes = len(node_neighbors_map)
    sum_node_degree = 0
    average_degree_neighbor = 0.
    if num_graph_nodes > 1000:
        # sample 500 nodes of the graph
        node_vals = np.array(list(node_neighbors_map.keys()))
        random_arr = np.random.choice(node_vals, 500)

        for rand_node in random_arr:
            node_neighbors = node_neighbors_map[rand_node]
            sum_node_degree += len(node_neighbors)
            if len(node_neighbors):
                average_degree_neighbor += (
                            sum([len(node_neighbors_map[n]) for n in node_neighbors]) / len(node_neighbors))
    else:
        for i in node_neighbors_map:
            node_neighbors = node_neighbors_map[i]
            sum_node_degree += len(node_neighbors)
            if len(node_neighbors):
                average_degree_neighbor += (
                            sum([len(node_neighbors_map[n]) for n in node_neighbors]) / len(node_neighbors))

    ratio = sum_node_degree / average_degree_neighbor
    return ratio


# Q5f (10 points)
'''
For each of the network files \texttt{enron.txt} (from your last problem set), \texttt{epinions.txt} (from your last problem set),
\texttt{higgs.txt} (spread of the announcement of the discovery of the Higgs boson in 2012), \texttt{slashdot.txt} (Slashdot Zoo social network as of November 2008),
\texttt{wikiTalk.txt} (Wikipedia Talk network up to 2008), and \texttt{youtube.txt} (YouTube social network as of 2012), estimate the Power Law exponent
(using your preferred approach as you justified in part c) and the Friendship Paradox Ratio for that network. Plot the Power Law exponent estimates as a
function of the Friendship Paradox Ratio estimates. You should plot the data for undirected and directed graphs on the same set of axes, so your plot
should contain exactly 6 points. Include a screenshot of your plot with your write-up and describe your results. 
'''

# Public Unit Tests (uncomment as you go)
assert (findExponentOLS(findNodeDegreesDirected("sample.txt")) - 0.907 < 0.001)
assert (abs(findExponentMML(findNodeDegreesDirected("sample.txt")) - 1.672) < 0.001)
assert(abs(directedFriendshipParadox("sample.txt") - 0.87) < 0.01)


def main():
    list_files = ["enron.txt", "epinions.txt", "higgs.txt", "slashdot.txt", "wikiTalk.txt", "youtube.txt"]
    list_exponents = []
    list_ratios = []
    for i in range(len(list_files)):
        print(f'fileName: {list_files[i]}')
        if list_files[i] == "youtube.txt":
            graph = findNodeDegreesUndirected(list_files[i])
            list_exponents.append(findExponentMML(graph))
            list_ratios.append(undirectedFriendshipParadox(list_files[i]))
        else:
            graph = findNodeDegreesDirected(list_files[i])
            list_exponents.append(findExponentMML(graph))
            list_ratios.append(directedFriendshipParadox(list_files[i]))

    plt.scatter(list_ratios, list_exponents)
    plt.xlabel('Friendship Paradox Ratio')
    plt.ylabel('Exponent')
    plt.title('Exponent vs. Paradox Ratio')
    plt.show()


if __name__ == '__main__':
    main()
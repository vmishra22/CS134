import math
import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from collections import deque


def create_torus_graph(dim):
    G = {}
    for i in range(1, dim + 1):
        for j in range(1, dim + 1):
            n = []
            n1 = (i - 1) * dim + j + 1
            if n1 > ((i - 1) * dim + dim):
                n1 -= dim
            n2 = (i - 1) * dim + j - 1
            if n2 <= (i - 1) * dim:
                n2 = (i - 1) * dim + dim
            n3 = (i - 1) * dim + j + dim
            if n3 > dim * dim:
                n3 = (i - 1) * dim + j + dim - (dim * dim)
            n4 = (i - 1) * dim + j - dim
            if n4 <= 0:
                n4 = (i - 1) * dim + j - dim + (dim * dim)

            G[(i - 1) * dim + j] = [n1, n2, n3, n4]
    return G


def experiment1Uniform(G, dim):
    additional_edges = {}
    for i in G:
        rand_value = i
        while rand_value == i:
            rand_value = int(np.random.randint(1, dim * dim + 1, 1))
        additional_edges[i] = rand_value

    for e in additional_edges:
        additional_node = additional_edges[e]
        G[e].append(additional_node)

    return G


def torus_distance_between_nodes(u, v, dim):
    rowlevel1 = int(math.ceil(u / dim))
    rowlevel2 = int(math.ceil(v / dim))
    vertical_distance = min((rowlevel1 % dim - rowlevel2 % dim) % dim, (rowlevel2 % dim - rowlevel1 % dim) % dim)
    horizontal_distance = min((u % dim - v % dim) % dim, (v % dim - u % dim) % dim)
    return horizontal_distance + vertical_distance


def binarySearch(arr, left, right, x):
    mid = -1
    while left <= right:

        mid = int(left + (right - left) / 2);

        midVal = float(arr[mid][1])
        # Check if x is present at mid
        if abs(midVal - x) < 1.0e-9:
            return mid

            # If x is greater, ignore left half
        elif midVal < x:
            left = mid + 1

        # If x is smaller, ignore right half
        else:
            right = mid - 1

    return mid


# Rejection sampling is used (took around 8 hours for 500X500 nodes),
# couldn't employ faster mechanism due to time constraints.
def experiment2_kleinberg(G, dim):
    kleinberg_edges = {}
    for i in G:
        node_found = False
        while not node_found:
            rand_value = i
            while rand_value == i:
                rand_value = int(np.random.randint(1, dim * dim + 1, 1))

            lattice_distance = torus_distance_between_nodes(i, rand_value, dim)
            kleinberg_prob = 1 / pow(lattice_distance, 2)

            pick_probability = float(np.random.uniform(0.0, 1.0, 1))
            if pick_probability <= kleinberg_prob:
                kleinberg_edges[i] = rand_value
                node_found = True

    # Now, add the kleinberg edges in graph
    for e in kleinberg_edges:
        additional_node = kleinberg_edges[e]
        G[e].append(additional_node)
    return G


def greedy_path_distance(G, u, v, dim):
    neighbors = G[u]
    dist_node_map = {}
    for nb in neighbors:
        if nb == v:
            return nb, torus_distance_between_nodes(u, nb, dim)
        distance_with_target = torus_distance_between_nodes(nb, v, dim)
        dist_node_map[nb] = distance_with_target

    sorted_by_value = sorted(dist_node_map.items(), key=lambda kv: kv[1])
    closest_node_to_target = sorted_by_value[0][0]
    shortest_dist_node = (closest_node_to_target, torus_distance_between_nodes(u, closest_node_to_target, dim))
    return shortest_dist_node


def greedy_path_distance_with_lookahead(G, u, v, dim):
    neighbors = G[u]
    dist_node_map = {}
    for nb in neighbors:
        if nb == v:
            return nb, torus_distance_between_nodes(u, nb, dim)
        for nb2 in G[nb]:
            distance_with_target = torus_distance_between_nodes(nb2, v, dim)
            dist_node_map[(nb, nb2)] = distance_with_target

    sorted_by_value = sorted(dist_node_map.items(), key=lambda kv: kv[1])
    closest_node_to_target = sorted_by_value[0][0]
    shortest_dist_node = (closest_node_to_target[0], torus_distance_between_nodes(u, closest_node_to_target[0], dim))
    return shortest_dist_node


# Calculate average and max distance
def calculate_avg_and_max_dist(G, dim, random_nodes, experiment_type):
    num_pairs = len(random_nodes)
    sum_distances = 0
    max_distance = 0
    index = 0
    for p1 in random_nodes:
        index += 1
        origin = p1[0]
        target = p1[1]
        nb = origin
        distance_target = 0
        while nb != target:
            step_node = greedy_path_distance(G, nb, target, dim)
            nb = step_node[0]
            distance_target += step_node[1]
        sum_distances += distance_target
        if distance_target > max_distance:
            max_distance = distance_target
    avg_distances = sum_distances / num_pairs
    print(f'Experiment: {experiment_type}, Average distance: {avg_distances}, Maximum distance: {max_distance}')


# Calculate average and max distance with lookahead
def calculate_avg_and_max_dist_with_lookahead(G, dim, random_nodes, experiment_type):
    num_pairs = len(random_nodes)
    sum_distances = 0
    max_distance = 0
    index = 0
    for p1 in random_nodes:
        index += 1
        origin = p1[0]
        target = p1[1]
        nb = origin
        distance_target = 0
        while nb != target:
            step_node = greedy_path_distance_with_lookahead(G, nb, target, dim)
            nb = step_node[0]
            distance_target += step_node[1]

        sum_distances += distance_target
        if distance_target > max_distance:
            max_distance = distance_target
    avg_distances = sum_distances / num_pairs
    print(f'Experiment: {experiment_type}, Average distance: {avg_distances}, Maximum distance: {max_distance}')


def main():
    dim = 500
    G1 = create_torus_graph(dim)
    G1Uniform = experiment1Uniform(G1, dim)
    G2 = create_torus_graph(dim)
    G2Kleinberg = experiment2_kleinberg(G2, dim)

    num_pairs = 10000
    random_nodes = []
    for i in range(num_pairs):
        rand_value1 = int(np.random.randint(1, dim * dim + 1, 1))
        rand_value2 = rand_value1
        while rand_value2 == rand_value1:
            rand_value2 = int(np.random.randint(1, dim * dim + 1, 1))
        random_nodes.append((rand_value1, rand_value2))

    calculate_avg_and_max_dist(G1Uniform, dim, random_nodes, 'Uniform Selection')
    calculate_avg_and_max_dist(G2Kleinberg, dim, random_nodes, 'Kleinberg Selection')
    calculate_avg_and_max_dist_with_lookahead(G1Uniform, dim, random_nodes, 'Uniform Selection with Lookahead')
    calculate_avg_and_max_dist_with_lookahead(G2Kleinberg, dim, random_nodes, 'Kleinberg Selection with Lookahead')


if __name__ == '__main__': main()

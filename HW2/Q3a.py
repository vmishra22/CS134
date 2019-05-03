import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from collections import deque
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path

def main():
    N=20
    x_values = np.logspace(0.1, 2, N, endpoint=True)
    xList = x_values.tolist()
    yList = []
    for i in xList:
        print(f'x: {i}')
        z1 = np.float64(np.exp(i)).item()
        y1 = i * (z1 - 0.5) + (z1 - 11 + z1 * np.float64(np.log(10)).item())
        yList.append(np.float64(np.log(y1)).item())

    plt.ylabel('ln(f(d))')
    plt.xlabel('ln(d)')
    plt.title(f'ln(d) vs ln(f(d)')
    plt.plot(xList, yList, linestyle='dashed', marker='o')
    plt.show()

if __name__ == '__main__': main()
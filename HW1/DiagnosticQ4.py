import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from collections import deque
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path


def countMax(L):
    print(L)
    curMax = L[0]
    counter = 1
    for x in L[1:len(L)]:
        if curMax < x:
            curMax = x
            counter = 1
        elif curMax == x:
            counter += 1

    print(f'Max count: {counter}')

def main():
    # Enter the list by each integer seperated with a space
    list1 = [int(x) for x in input().split()]
    countMax(list1)


if __name__ == '__main__':
    main()
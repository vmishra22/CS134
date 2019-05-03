import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from collections import deque
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path


def balls_bins_simulation(p, nTrials, nBalls):
    trialIndex = 0
    listGreen = []
    while trialIndex < nTrials:
        nR = 200
        nG = 100
        nTotal = nR + nG
        randVal = np.random.uniform(0.0, 1.0, nBalls)
        i = 0
        while nTotal < nBalls:
            prRed = (pow(nR, p) / (pow(nR, p) + pow(nG, p)))
            prGreen = (pow(nG, p) / (pow(nR, p) + pow(nG, p)))
            if prRed <= prGreen:
                if randVal[i] <= prRed:
                    nR += 1
                else:
                    nG += 1
            else:
                if randVal[i] <= prGreen:
                    nG += 1
                else:
                    nR += 1

            nTotal = nR + nG
            i += 1

        listGreen.append(nG)
        trialIndex += 1
        print(f'trialIndex: {trialIndex}')

    averageGreen = sum(listGreen) / len(listGreen)
    minGreen = min(listGreen)
    maxGreen = max(listGreen)
    print(f'Total balls: {nBalls}, Min:{minGreen}, Max:{maxGreen}, Average{averageGreen}')

def main():
    p = 2
    #balls_bins_simulation(p, 1000, 10000)
    balls_bins_simulation(p, 1000, 1000000)


if __name__ == '__main__':
    main()

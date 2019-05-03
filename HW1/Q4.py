import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from collections import deque
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path


def part1():
    nTrials = 100
    trialIndex = 0
    listX = []
    while trialIndex < nTrials:
        nR = 1
        nB = 1
        nTotal = nR + nB
        while nTotal < 1.0e6:
            randVal = np.random.uniform(0.0, 1.0, 1)
            prRed = (nR / nTotal)
            prBlue = (nB / nTotal)
            if prRed <= prBlue:
                if randVal <= prRed:
                    nR += 1
                else:
                    nB += 1
            else:
                if randVal <= prBlue:
                    nB += 1
                else:
                    nR += 1
            nTotal = nR + nB

        listX.append(min(nR, nB))
        trialIndex += 1

    xValues = np.sort(listX)
    yValues = np.arange(1, len(xValues) + 1) / len(xValues)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title(f'First Experiment with p = 1')
    plt.plot(xValues, yValues)
    plt.show()


def part2(p):
    nTrials = 5000
    trialIndex = 0
    listX = []
    while trialIndex < nTrials:
        nR = 1
        nB = 1
        nTotal = nR + nB
        randVal = np.random.uniform(0.0, 1.0, 1000000)
        i=0
        while nTotal < 1.0e6:
            prRed = (pow(nR, p) / (pow(nR, p) + pow(nB, p)))
            prBlue = (pow(nB, p) / (pow(nR, p) + pow(nB, p)))
            if prRed <= prBlue:
                if randVal[i] <= prRed:
                    nR += 1
                else:
                    nB += 1
            else:
                if randVal[i] <= prBlue:
                    nB += 1
                else:
                    nR += 1

            nTotal = nR + nB
            i += 1

        listX.append(min(nR, nB))
        trialIndex += 1
        print(f'trialIndex: {trialIndex}')

    xValues = np.sort(listX)
    yValues = np.arange(1, len(xValues) + 1) / len(xValues)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title(f'Second Experiment with p = {p}')
    plt.plot(xValues, yValues)
    plt.show()


def main():
    part1()
    p = 2
    part2(p)
    p = 0.5
    part2(p)


if __name__ == '__main__':
    main()

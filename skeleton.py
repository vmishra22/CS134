# from numpy import *
from __future__ import division

from collections import Counter

import cluster_funs  ## IMPORTANT imports our helper functions
import numpy as np
import pandas as pd
from math import sqrt
from scipy.cluster.vq import kmeans, kmeans2, whiten, vq
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import plot, show
import scipy

## Mapping Tools
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon


def ratio(A, clusters):
    N = clusters.shape[0]
    unique, counts = np.unique(clusters, return_counts=True)
    clusters_dict_count = dict(zip(unique, counts))

    clusters = clusters.flatten()

    grand_sum = 0.0
    for i in unique:
        index = np.where(clusters == i)
        index1 = np.where(clusters != i)
        Nc = clusters_dict_count[i]
        r1 = float(Nc / N)
        r2 = float(np.power(Nc, 2) - Nc) / float(2 * Nc * (N - Nc))
        if r2 == 0.0:
            continue
        A_i_i = (A[index[0], :][:, index[0]]).sum()
        A_i_j = (A[index[0], :][:, index1[0]]).sum()
        A_j_i = (A[index1[0], :][:, index[0]]).sum()
        grand_sum += (r1 / r2) * (A_i_i / (A_i_j + A_j_i))

    return grand_sum


def unit_test_ratio():
    A_test = pd.read_csv('./Data/A_for_ratio_unit_test.csv', delimiter=',').as_matrix()
    np.fill_diagonal(A_test, 0)
    clusters_test = pd.read_csv('./Data/clusts_for_ratio_unit_test.csv', delimiter=',').as_matrix()

    N = 1000
    # A_test = np.random.randint(low=1, high = N, size=N**2).reshape((N,N))
    # np.fill_diagonal(A_test, 0)
    # clusters_test = np.random.randint(1, 4, size=N)

    print(ratio(A_test, clusters_test))

    if (abs(ratio(A_test, clusters_test) - 0.995829043348) < 0.0001):
        print("Unit Test for Ratio passed");
        return
    else:
        print(
            "Unit Test for Ratio failed. Answer should be near 1 in this unit test. Specifically, ratio should be 0.995829043348")


def compute_euclid_distance(a, mu_points):
    return np.array([np.linalg.norm(a - mu_points[i]) for i in range(len(mu_points))])


def kMeanAlgorithm(X, k, map_k):
    Xmin = np.amin(X, axis=0)
    Xmax = np.amax(X, axis=0)

    mu_points = np.zeros((k, 2), dtype=np.float64)

    # TODO
    # Initialize mu values
    for i in range(k):
        mu_points[i] = [np.random.uniform(Xmin[0], Xmax[0], 1), np.random.uniform(Xmin[1], Xmax[1], 1)]

    previous_mu_points = mu_points.copy()

    convergence = False
    while not convergence:
        label_arr = {}
        for i in range(X.shape[0]):
            distances = compute_euclid_distance(X[i, :], mu_points)
            minMuIndex = np.argmin(distances)
            if minMuIndex not in label_arr:
                label_arr[minMuIndex] = []
            label_arr[minMuIndex].append(i)

        mu_points_test = mu_points.copy()
        ptt = np.linalg.norm(X - mu_points_test[:, None], axis=2)
        n_label_arr = np.argmin(np.linalg.norm(X - mu_points_test[:, None], axis=2), axis=0)

        for i in label_arr.keys():
            mu_points[i] = np.array(
                [np.sum(X[m][0] for m in label_arr[i]), np.sum(X[m][1] for m in label_arr[i])]) / np.float(
                len(label_arr[i]))

        comp_arr = np.isclose(previous_mu_points, mu_points)
        if comp_arr.all():
            convergence = True
        else:
            previous_mu_points = mu_points.copy()

    if map_k == k:
        color = ['black', 'blue', 'red', 'green']
        x_values = np.array([[X[m][0] for m in label_arr[i]] for i in label_arr.keys()])
        y_values = np.array([[X[m][1] for m in label_arr[i]] for i in label_arr.keys()])
        plt.figure(1)
        area = np.pi * 3
        for ki in range(k):
            plt.scatter(x_values[ki], y_values[ki], s=area, color=color[ki], alpha=0.5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Scatter plot for elbow value k=4')
        plt.show()

    final_distances = np.zeros(X.shape[0], dtype=np.float64)
    fi = 0
    for i in label_arr.keys():
        for m in label_arr[i]:
            xDiff = float(X[m][0]) - float(mu_points[i][0])
            yDiff = float(X[m][1]) - float(mu_points[i][1])
            final_distances[fi] = np.power(xDiff, 2) + np.power(yDiff, 2)
            fi += 1
    return np.sum(final_distances)


if __name__ == '__main__':
    ###############################################################
    #####################     Question 1     ######################
    ###############################################################

    ## YOUR WORK HERE ##
    ####################
    k = 7
    k_values = np.array(range(k)) + 1

    X_t = np.genfromtxt(f'./Data/toydata.csv', delimiter=',', skip_header=1, usecols=(0, 1))
    G_Sum_arr = np.array([kMeanAlgorithm(X_t, ik, 4) for ik in k_values])
    plt.figure(2)
    plt.plot(k_values, G_Sum_arr)
    plt.xlabel('k')
    plt.ylabel('Cluster sum of squares')
    plt.title('Sum vs k')
    plt.show()

    # Part 4b
    mun_vote_df = pd.read_csv('./data/mun_vote.csv', delimiter=',')
    mun_ethn_df = pd.read_csv('./data/mun_ethn.csv', delimiter=',')
    vote_arr = pd.DataFrame.as_matrix(mun_vote_df)
    ethn_arr = pd.DataFrame.as_matrix(mun_ethn_df)
    t_vote_arr = vote_arr[:, 1:]
    t_vote_arr = whiten(t_vote_arr)
    t_ethn_arr = ethn_arr[:, 1:]
    t_ethn_arr = whiten(t_ethn_arr)

    k = 16
    k_values = np.array(range(k)) + 1
    clx_votes = np.zeros((t_vote_arr.shape[0], k), dtype=np.int)
    clx_ethn = np.zeros((t_ethn_arr.shape[0], k), dtype=np.int)
    G_sum_votes = np.zeros(k, dtype=np.float64)
    G_sum_ethn = np.zeros(k, dtype=np.float64)
    for ki in k_values:
        centroids_votes, _ = kmeans(t_vote_arr, ki)
        centroids_ethn, _ = kmeans(t_ethn_arr, ki)
        clx_votes[:, ki - 1], _ = vq(t_vote_arr, centroids_votes)
        clx_ethn[:, ki - 1], _ = vq(t_ethn_arr, centroids_ethn)

        sum = 0.
        for zi in range(ki):
            indicesV = np.where(clx_votes[:, ki - 1] == zi)
            for vi in indicesV[0]:
                G_sum_votes[ki - 1] += np.sum(np.power(t_vote_arr[vi] - centroids_votes[zi], 2))

            indicesE = np.where(clx_ethn[:, ki - 1] == zi)
            for ei in indicesE[0]:
                G_sum_ethn[ki - 1] += np.sum(np.power(t_ethn_arr[ei] - centroids_ethn[zi], 2))

    plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.plot(k_values, G_sum_votes, 'o-')
    plt.title('Elbow graphs')
    plt.ylabel('Sum_Squares_Votes')

    plt.subplot(2, 1, 2)
    plt.plot(k_values, G_sum_ethn, '.-')
    plt.xlabel('k')
    plt.ylabel('Sum_Squares_Ethnicites')
    plt.show()

    label_arr_k_6_vote = {}
    for zi in range(6):
        indicesV = np.where(clx_votes[:, 5] == zi)
        label_arr_k_6_vote[zi] = (vote_arr[indicesV, 0]).astype(int)

    label_arr_k_5_eth = {}
    for zi in range(5):
        indicesE = np.where(clx_ethn[:, 4] == zi)
        label_arr_k_5_eth[zi] = (ethn_arr[indicesE, 0]).astype(int)

    lat_lon_df = pd.read_csv('./data/mun_latlon.csv', delimiter=',')
    lat_lon_arr = pd.DataFrame.as_matrix(lat_lon_df)

    x_vote_cluster_6_values = []
    y_vote_cluster_6_values = []

    x_vote_cluster_6_values = np.array(
        [[lat_lon_arr[(np.where(lat_lon_arr[:, 0] == m))[0]][0][1] for m in label_arr_k_6_vote[i][0]] for i in
         label_arr_k_6_vote.keys()])
    y_vote_cluster_6_values = np.array(
        [[lat_lon_arr[(np.where(lat_lon_arr[:, 0] == m))[0]][0][2] for m in label_arr_k_6_vote[i][0]] for i in
         label_arr_k_6_vote.keys()])

    x_ethn_cluster_5_values = []
    y_ethn_cluster_5_values = []

    x_ethn_cluster_5_values = np.array(
        [[lat_lon_arr[(np.where(lat_lon_arr[:, 0] == m))[0]][0][1] for m in label_arr_k_5_eth[i][0]] for i in
         label_arr_k_5_eth.keys()])
    y_ethn_cluster_5_values = np.array(
        [[lat_lon_arr[(np.where(lat_lon_arr[:, 0] == m))[0]][0][2] for m in label_arr_k_5_eth[i][0]] for i in
         label_arr_k_5_eth.keys()])

    color = ['orange', 'blue', 'red', 'green', 'cyan', 'yellow']
    plt.figure(4)

    plt.subplot(2, 1, 1)
    area = np.pi * 3
    plt.title('Cluster for Votes k=6')
    for ki in range(6):
        plt.scatter(x_vote_cluster_6_values[ki], y_vote_cluster_6_values[ki], s=area, color=color[ki], alpha=0.5)

    plt.subplot(2, 1, 2)
    plt.title('Cluster for Ethnicities k=5')
    for ki in range(5):
        plt.scatter(x_ethn_cluster_5_values[ki], y_ethn_cluster_5_values[ki], s=area, color=color[ki], alpha=0.5)
    plt.xlabel('Lattitude')
    plt.ylabel('Longitude')
    plt.show()

    #####################################################
    ##  EXAMPLE of COLORING MUNICIPALITIES OF COLOMBIA ##
    #####################################################

    # Generate the empty map of the area around Colombia
    map1 = Basemap(llcrnrlon=-65, llcrnrlat=-6, urcrnrlon=-80, urcrnrlat=15)

    # Load the shape files.
    map1.readshapefile('./Data/COL_adm2', name='municipalities', drawbounds=True)

    # The following checks the shapefile data and makes a list such that we can
    # reference each municipality by its index. Unfortunately, the shapefiles do NOT
    # have the actual municipality codes that everyone else uses -- they just have indices
    # from 1 to 1065. We have provided a csv file lookup table to convert back and forth
    # NOTE some lookups are missing. If you encounter one, then just leave that municipality uncolored

    mun_idx1 = []
    for shape_dict in map1.municipalities_info:
        # We will be using the column ID_2 to reference municipalities (see in the shapefile csv)
        mun_idx1.append(shape_dict['ID_2'])

    # Read CSV to convert from shapefile ID's to municipality codes ("Cod.Municipio")
    codebook = pd.read_csv('./Data/COL_adm2_Conversion.csv', delimiter=',')

    # Convert to dictionary for quick lookups
    codebook_dict = dict(zip(list(codebook['Cod.Municipio']), list(codebook['ID_2'])))

    # Here's how we would color the municipality Leticia (ID_2==4; Cod.Municipio==91001) red:
    ax = plt.gca()  # Get current plot axes
    # seg1 = map1.municipalities[mun_idx.index(codebook_dict[91001])]  ## THIS is the important part

    for i in label_arr_k_6_vote.keys():
        for m in label_arr_k_6_vote[i][0]:
            if m in codebook_dict:
                seg1 = map1.municipalities[mun_idx1.index(codebook_dict[m])]
                poly1 = Polygon(seg1, facecolor=color[i], edgecolor=color[i])
                ax.add_patch(poly1)  # Add the colored in polygon to our map

    plt.title('Cluster colors for k=6 Votes over map')
    plt.show()
    plt.close()

    # Generate the empty map of the area around Colombia
    map2 = Basemap(llcrnrlon=-65, llcrnrlat=-6, urcrnrlon=-80, urcrnrlat=15)

    # Load the shape files.
    map2.readshapefile('./Data/COL_adm2', name='municipalities', drawbounds=True)
    mun_idx2 = []
    for shape_dict in map2.municipalities_info:
        # We will be using the column ID_2 to reference municipalities (see in the shapefile csv)
        mun_idx2.append(shape_dict['ID_2'])

    ax2 = plt.gca()
    for i in label_arr_k_5_eth.keys():
        for m in label_arr_k_5_eth[i][0]:
            if m in codebook_dict:
                seg2 = map2.municipalities[mun_idx2.index(codebook_dict[m])]
                poly2 = Polygon(seg2, facecolor=color[i], edgecolor=color[i])
                ax2.add_patch(poly2)  # Add the colored in polygon to our map

    plt.title('Cluster colors for k=5 Ethnicities over map')
    plt.show()
    plt.close()
    ## Part 1.C
    unit_test_ratio()
    A_vote = np.zeros((vote_arr.shape[0], vote_arr.shape[0]))
    municipalities_node_idx = (vote_arr[:, 0]).astype(int)
    np.sort(municipalities_node_idx)
    nodes_calls_df = pd.read_csv('./data/mun_mun_social_edgelist_header_symm_859.csv', delimiter=',')
    node_calls_arr = pd.DataFrame.as_matrix(nodes_calls_df)
    for ci in node_calls_arr:
        idx1 = np.where(municipalities_node_idx == ci[0])
        idx2 = np.where(municipalities_node_idx == ci[1])
        dx1 = idx1[0]
        dx2 = idx2[0]
        A_vote[dx1[0]][dx2[0]] = ci[2]
    np.fill_diagonal(A_vote, 0)

    nTrials = 20
    votes_ratios = np.zeros(nTrials)
    ethn_ratios = np.zeros(nTrials)
    for tr in range(nTrials):
        print(f'trial: {tr}')
        centroids_votes, _ = kmeans(t_vote_arr, 6)
        centroids_ethn, _ = kmeans(t_ethn_arr, 5)
        clx_votes_k6, _ = vq(t_vote_arr, centroids_votes)
        clx_ethn_k5, _ = vq(t_ethn_arr, centroids_ethn)
        clusters_vote_arr = clx_votes_k6 + 1
        clusters_ethn_arr = clx_ethn_k5 + 1
        votes_ratios[tr] = ratio(A_vote, clusters_vote_arr)
        ethn_ratios[tr] = ratio(A_vote, clusters_ethn_arr)

    print(f'Ratio for 6 vote clusters: {np.mean(votes_ratios)}, '
          f'Ratio for 5 ethnicities clusters:{np.mean(ethn_ratios)}')


    ###############################################################
    #####################     Question 2     ######################
    ###############################################################

    ### PART 2.A ###
    ###############################################################

    # Load the list of latitude and longitude for each municipality
    mun_lonlat_DF = pd.read_csv('./Data/mun_latlon.csv', delimiter=',')

    # Cast to Numpy Matrix
    cities_lonlat = mun_lonlat_DF[['lon', 'lat']].as_matrix()

    ## YOUR WORK HERE ##
    ####################
    # 5a - i
    mun_geodist_DF = pd.read_csv('./Data/mun_pairwise_geo_dist.csv', delimiter=',')
    cities_distance_matrix = pd.DataFrame.as_matrix(mun_geodist_DF)
    Y, e = cluster_funs.cmdscale(cities_distance_matrix)
    plt.figure(5, figsize=(10, 6), dpi=100)
    plt.scatter(cities_lonlat[:, 0], cities_lonlat[:, 1], c='b', s=30, marker='x', alpha=0.5)
    plt.scatter(Y[:, 0], Y[:, 1], c='r', s=30, marker='o', alpha=0.5)
    plt.title('Map of municipalities with CMD scaling')
    plt.savefig("OriginalMapCMDscaling.png")
    plt.show()
    plt.close()

    # 5a - ii
    _, Z, _ = cluster_funs.procrustes(cities_lonlat, Y)
    plt.figure(6)
    plt.scatter(cities_lonlat[:, 0], cities_lonlat[:, 1], c='b', s=30, marker='x', alpha=0.5)
    plt.scatter(Z[:, 0], Z[:, 1], c='r', s=30, marker='|', alpha=0.5)
    plt.title('Map of municipalities with CMD scaling after Procrustes procedure')
    plt.savefig("OriginalMapCMDscalingProcrustes.png")
    plt.show()
    plt.close()

    # 5b
    Distance_Matrix = np.zeros((vote_arr.shape[0], vote_arr.shape[0]))
    for ci in node_calls_arr:
        idx1 = np.where(municipalities_node_idx == ci[0])
        idx2 = np.where(municipalities_node_idx == ci[1])
        dx1 = idx1[0]
        dx2 = idx2[0]
        Distance_Matrix[dx1[0]][dx2[0]] = 1. / (ci[2] + 1.)
    np.fill_diagonal(Distance_Matrix, 0)
    Y1, e1 = cluster_funs.cmdscale(Distance_Matrix)
    centroids_cmd, _ = kmeans(Y1, 6)
    clx_cmd_k6, _ = vq(Y1, centroids_cmd)
    color = ['black', 'blue', 'red', 'green', 'cyan', 'yellow']

    lat_lon_k_6 = {}
    for kc in range(6):
        indicesKc = np.where(clx_cmd_k6[:] == kc)
        lat_lon_k_6[kc] = cities_lonlat[indicesKc[0], :]

    plt.figure(7, figsize=(15, 10), dpi=100)
    area = np.pi * 3
    for kc in range(6):
        plt.scatter(lat_lon_k_6[kc][:, 0], lat_lon_k_6[kc][:, 1], s=area, color=color[kc], alpha=0.5)
    plt.xlabel('lattitude')
    plt.ylabel('longitude')
    plt.title('Cluster plot for lat-lon k=6')
    plt.show()
    plt.close()

    map3 = Basemap(llcrnrlon=-65, llcrnrlat=-6, urcrnrlon=-80, urcrnrlat=15)

    # Load the shape files.
    map3.readshapefile('./Data/COL_adm2', name='municipalities', drawbounds=True)
    mun_idx3 = []
    for shape_dict in map3.municipalities_info:
        mun_idx3.append(shape_dict['ID_2'])

    ax3 = plt.gca()
    for kc in range(6):
        indicesKc = np.where(clx_cmd_k6[:] == kc)
        for kk in indicesKc[0]:
            seg3 = map3.municipalities[kk]
            poly3 = Polygon(seg3, facecolor=color[kc], edgecolor=color[kc])
            ax3.add_patch(poly3)
    plt.title('Cluster colors for k=6 high-pairwise-call over map')
    plt.show()

    # 5.(c)
    cluseters_cmd_k6_high_pairwise = clx_cmd_k6 + 1
    ratio_for_k_6_high_pairwise = ratio(Distance_Matrix, cluseters_cmd_k6_high_pairwise)
    print(f'Ratio for 6 vote clusters geographically: {ratio_for_k_6_high_pairwise}')


    A_vote_share_distance_matrix = squareform(pdist(t_vote_arr))
    ratio_for_A_vote_share_distance = ratio(A_vote_share_distance_matrix, cluseters_cmd_k6_high_pairwise)
    print(f'Ratio for 6 clusters vote share: {ratio_for_A_vote_share_distance}')
from numpy.linalg import matrix_power

import voter_funs  # this imports the helper functions & unit testing functions that have been provided for you
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd
from plotnine import *
from collections import deque, Counter
import csv


############################################################################
############################# Functions to Compose #########################
############################################################################

def voter_mat(A, opinions, t):
    ## INPUTS ##
    # A, a symmetric logical (Boolean) N x N adjacency np.array of graph G with all self loops (diagonal) == 1
    # opinions, a logical (Boolean) 1 x N np.array where opinions[i] == True means that node i holds opinion "True" at
    # time 0 t, a positive integer that represents the number of iterations of the voter algorithm to run

    ## OUTPUTS ##
    # opinions_mat, the logical (Boolean) (t+1) x N np.array where opinions[i]==True means that node i holds opinion
    # "True" at time delta timestep of the voter algorithm

    opinions_mat = np.array([[False] * A.shape[0]] * (t + 1))

    ###########################################
    #########  ** YOUR WORK HERE ** ###########
    ###########################################
    rows = A.shape[0]
    cols = A.shape[1]
    M = np.zeros((rows, cols), dtype=float)

    for i in range(0, rows):
        countN = Counter(A[i])
        prob = float((1. / countN[True]))
        M[i] = np.where(A[i] == True, prob, 0.)

    opinions_mat[0] = opinions
    ti = 1
    while ti <= t:
        X = np.dot(M, opinions_mat[ti - 1])
        rand_list = np.random.uniform(0.0, 1.0, cols)
        comp = np.less_equal(rand_list, X)

        # for ind, x in enumerate(comp):
        #     if x:
        #         opinions_mat[ti][ind] = True
        #     else:
        #         opinions_mat[ti][ind] = False

        opinions_mat[ti] = [np.bitwise_or(opinions_mat[ti - 1][ind], True) if x
                            else np.bitwise_and(opinions_mat[ti - 1][ind], False)
                            for ind, x in enumerate(comp)]
        ti += 1

    return opinions_mat


#############################################
# mu_1: Compute fraction of opinions that == 1 (i.e. mean(opinions))
def compute_opinion_fraction(opinions):
    ## INPUTS ##
    # opinions, a logical (Boolean) 1 x N np.array where opinions[i] == True means that node i holds opinion "True" at time 0

    ## OUTPUTS ##
    # The float mu_1 (see pset writeup)

    if ((not opinions.ndim == 1) | (not (opinions.dtype == bool))):
        return ("ERROR: node opinions vector must be a row vector (1d np.array) all in (T,F)**N space")

    ###########################################
    #########  ** YOUR WORK HERE ** ###########
    ###########################################
    p = 0.0
    countN = Counter(opinions)
    p = float(countN[True] / len(opinions))
    if p > 1.0 or p < 0.0:
        print(f'mu_1: countN[True]: {countN[True]}, opinions: {len(opinions)}', p)

    return p


#############################################
# mu_2: Compute probability that two randomly selected nodes have the same opinion
def compute_agreement_fraction(opinions):
    ## INPUTS ##
    # opinions, a logical (Boolean) 1 x N np.array where opinions[i] == True means that node i holds opinion "True" at time 0

    ## OUTPUTS ##
    # The float mu_2 (see pset writeup)

    ###########################################
    #########  ** YOUR WORK HERE ** ###########
    ###########################################
    p = 0.0
    countN = Counter(opinions)
    tO = len(opinions)
    pT = countN[True]
    pF = countN[False]
    p = (float(pT * (pT - 1) / 2) + float(pF * (pF - 1) / 2)) / float(tO * (tO - 1) / 2)
    if p > 1.0 or p < 0.0:
        print(
            f'mu_2 : countN[True]: {countN[True]}, countN[False]: {countN[False]}, opinions: {len(opinions)}, p = {p}')

    return p


#############################################
# mu_3: Compute fraction of edges where source and target agree
def compute_neighbor_agreement_fraction(A, opinions):
    ## INPUTS ##
    # A, a symmetric logical (Boolean) N x N adjacency np.array of graph G with all self loops (diagonal) == 1
    # opinions, a logical (Boolean) 1 x N np.array where opinions[i] == True means that node i holds opinion "True" at time 0

    ## OUTPUTS ##
    # The float mu_3 (see pset writeup)

    ###########################################
    #########  ** YOUR WORK HERE ** ###########
    ###########################################
    p = 0.0
    rows = A.shape[0]
    cols = A.shape[1]
    E = {}
    listNeighbors = [(i, j) for i in range(0, rows) for j in range(0, cols) if (i is not j) and (A[i, j])]
    if len(listNeighbors) is 0:
        return np.NaN
    else:
        for (i, j) in listNeighbors:
            if i not in E:
                E[i] = []
            E[i].append(opinions[j])

    totalPairs = 0
    totalTPairs = 0
    for ol in E:
        mL = [i for i in range(0, len(E[ol])) if opinions[ol] == E[ol][i]]
        totalPairs += len(E[ol])
        totalTPairs += len(mL)

    p = float(totalTPairs / totalPairs)

    return p


#############################################
# mu_4 Compute square of euclidean distance (norm) to a different opinions vector
def compute_euclid_norm_to_other_opinions(opinions, other_opinions):
    ## INPUTS ##
    # opinions, a logical (Boolean) 1 x N np.array where opinions[i] == True means that node i holds opinion "True" at time 0
    # other_opinions, a logical (Boolean) 1 x N np.array where opinions[i] == True means that node i holds opinion "True" at time 0

    ## OUTPUTS ##
    # The float mu_4 (see pset writeup)

    ###########################################
    #########  ** YOUR WORK HERE ** ###########
    ###########################################
    p = 0.0
    size = len(opinions)
    diff = np.bitwise_xor(opinions, other_opinions)
    s = np.sum(np.array([i * i for i in diff]))
    p = float(s / size)
    return p


if __name__ == '__main__':
    ####################################################################################
    ############## Unit Testing your voter_mat() and mu functions ######################
    #############  # !! NO NEED to change code in this section !!  #####################

    # Run unit test for part (a)
    voter_funs.unit_test_voter_mat()

    # Run unit tests for part (b)
    voter_funs.unit_test_mu1_mu2_mu3()

    voter_funs.unit_test_mu_4()

    ####################################################################################
    ######################## Simulations and Analysis ##################################
    #############  # !! NO NEED to change code in this section !!#######################

    # Run the simulations for part (c)
    # Search over many parameter combinations, (modified) Erdos-Renyi edition:
    N = 60
    A_gen_fun = voter_funs.make_symmetric_ER_w_all_self_loops  # Erdos-Renyi
    A_gen_fun_param_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    opinion_density_vec = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    t = 64
    matrix_draws_per_A_param = 15
    voter_runs_per_A_t = 5
    ts_to_plot = np.array([0, 4, 16, 64])

    if any((ts_to_plot > t) | (ts_to_plot < 0)): raise ValueError("ts_to_plot must be in the domain of t!")

    print("Part (c) INITIALIZATION OF PARAMETER SEARCH COMPLETE")

    # outputs_list = voter_funs.run_voter_over_params(N=N,
    #                                                 A_gen_fun=A_gen_fun,  # Our modified Erdos-Renyi network generator
    #                                                 A_gen_fun_param_vec=A_gen_fun_param_vec,
    #                                                 opinions0_densities_vec=opinion_density_vec,
    #                                                 t=t,
    #                                                 matrix_draws_per_A_param=matrix_draws_per_A_param,
    #                                                 voter_runs_per_A_t=voter_runs_per_A_t)

    print("Part (c) SIMULATIONS COMPLETE. GENERATING PLOTS AND SAVING THEM TO CURRENT DIRECTORY")

    # Convert output output means in our np.arrays to Pandas Data Frames for plotting
    # We are only going to plot snapshots of the means at particular timesteps i.e. ts_to_plot,
    # output_opin_frac_means = outputs_list[3][:, :, ts_to_plot]
    # output_agreement_frac_means = outputs_list[4][:, :, ts_to_plot]
    # output_neighbor_agreement_frac_means = outputs_list[5][:, :, ts_to_plot]
    #
    # # Declare iterables i.e. the indices we looped over to create this array
    # iterables = [A_gen_fun_param_vec, opinion_density_vec, ts_to_plot]
    #
    # # Generate Pandas index from iterables
    # index = pd.MultiIndex.from_product(iterables, names=['Erdos-Renyi_Density_p', 'init_opinions_P(True)', 'time'])
    #
    # # Generate long-formatted data frames that are ready to plot in the Grammar of Graphics style
    # output_opin_frac_means_MELT = pd.DataFrame(output_opin_frac_means.reshape(np.prod(output_opin_frac_means.shape)),
    #                                            index=index)
    # output_opin_frac_means_MELT.reset_index(inplace=True)
    # output_opin_frac_means_MELT.rename(columns={output_opin_frac_means_MELT.columns[3]: 'value'}, inplace=True)
    #
    # output_agreement_frac_means_MELT = pd.DataFrame(
    #     output_agreement_frac_means.reshape(np.prod(output_agreement_frac_means.shape)), index=index)
    # output_agreement_frac_means_MELT.reset_index(inplace=True)
    # output_agreement_frac_means_MELT.rename(columns={output_agreement_frac_means_MELT.columns[3]: 'value'},
    #                                         inplace=True)
    #
    # output_neighbor_agreement_frac_means_MELT = pd.DataFrame(
    #     output_neighbor_agreement_frac_means.reshape(np.prod(output_neighbor_agreement_frac_means.shape)), index=index)
    # output_neighbor_agreement_frac_means_MELT.reset_index(inplace=True)
    # output_neighbor_agreement_frac_means_MELT.rename(
    #     columns={output_neighbor_agreement_frac_means_MELT.columns[3]: 'value'}, inplace=True)
    #
    # output_agreement_vs_neighbor_agreement_MELT = output_agreement_frac_means_MELT.copy()
    # output_agreement_vs_neighbor_agreement_MELT['mu_3'] = output_neighbor_agreement_frac_means_MELT['value']
    # output_agreement_vs_neighbor_agreement_MELT.rename(
    #     columns={output_agreement_vs_neighbor_agreement_MELT.columns[3]: 'mu_2'}, inplace=True)
    #
    # # Similar, but let's make a data frame of mu_2 vs mu_3 by every combination of time, A matrix edge density, and initial opinions balance
    # iterables_alltime = iterables = [A_gen_fun_param_vec, opinion_density_vec, range((t + 1))]
    # index_alltime = pd.MultiIndex.from_product(iterables_alltime,
    #                                            names=['Erdos-Renyi_Density_p', 'init_opinions_P(True)', 'time'])
    #
    # output_agreement_frac_means_alltime = outputs_list[4]
    # output_agreement_frac_means_MELT_alltime = pd.DataFrame(
    #     output_agreement_frac_means_alltime.reshape(np.prod(output_agreement_frac_means_alltime.shape)),
    #     index=index_alltime)
    # output_agreement_frac_means_MELT_alltime.reset_index(inplace=True)
    # output_agreement_frac_means_MELT_alltime.rename(
    #     columns={output_agreement_frac_means_MELT_alltime.columns[3]: 'mu_2'}, inplace=True)
    #
    # output_neighbor_agreement_frac_means_alltime = outputs_list[5]
    # output_neighbor_agreement_frac_means_MELT_alltime = pd.DataFrame(
    #     output_neighbor_agreement_frac_means_alltime.reshape(
    #         np.prod(output_neighbor_agreement_frac_means_alltime.shape)), index=index_alltime)
    # output_neighbor_agreement_frac_means_MELT_alltime.reset_index(inplace=True)
    # output_neighbor_agreement_frac_means_MELT_alltime.rename(
    #     columns={output_neighbor_agreement_frac_means_MELT_alltime.columns[3]: 'value'}, inplace=True)
    #
    # output_agreement_frac_means_MELT_alltime['mu_3'] = output_neighbor_agreement_frac_means_MELT_alltime['value']
    #
    # # Produce and save plots
    # opin_frac_mean_ERsims_plot = (ggplot(output_opin_frac_means_MELT,
    #                                      aes(x="Erdos-Renyi_Density_p", y="init_opinions_P(True)", fill='value')) +
    #                               geom_tile() +  # makes a heatmap
    #                               ggtitle("Mean Opinion 0 to 1 (mu_1)") +
    #                               facet_wrap('~time', ncol=len(
    #                                   ts_to_plot)) +  # splits into different side-by-side plots -- one for each of the ts_to_plot
    #                               scale_y_continuous(breaks=opinion_density_vec) +  #
    #                               scale_x_continuous(breaks=A_gen_fun_param_vec) +  #
    #                               scale_fill_gradient(low="blue", high="yellow", limits=[0.1, 1]))
    # opin_frac_mean_ERsims_plot.save('Sims_plot_opin_frac_mean', width=16, height=4)
    #
    # agreement_frac_means_ERsims_plot = (ggplot(output_agreement_frac_means_MELT,
    #                                            aes(x="Erdos-Renyi_Density_p", y="init_opinions_P(True)",
    #                                                fill='value')) +
    #                                     geom_tile() +
    #                                     ggtitle("Probability that Two Nodes Agree (mu_2)") +
    #                                     facet_wrap('~time', ncol=len(ts_to_plot)) +
    #                                     scale_y_continuous(breaks=opinion_density_vec) +
    #                                     scale_x_continuous(breaks=A_gen_fun_param_vec) +  #
    #                                     scale_fill_gradient(low="blue", high="yellow", limits=[0.1, 1]))
    # agreement_frac_means_ERsims_plot.save('Sims_plot_agreement_frac_means', width=16, height=4)
    #
    # neighbor_agreement_frac_means_ERsims_plot = (ggplot(output_neighbor_agreement_frac_means_MELT,
    #                                                     aes(x="Erdos-Renyi_Density_p", y="init_opinions_P(True)",
    #                                                         fill='value')) +
    #                                              geom_tile() +
    #                                              ggtitle("Probability that Pair of Neighbors Agree (mu_3)") +
    #                                              facet_wrap('~time', ncol=len(ts_to_plot)) +
    #                                              scale_y_continuous(breaks=opinion_density_vec) +
    #                                              scale_x_continuous(breaks=A_gen_fun_param_vec) +  #
    #                                              scale_fill_gradient(low="blue", high="yellow", limits=[0.1, 1]))
    # neighbor_agreement_frac_means_ERsims_plot.save('Sims_plot_neighbor_agreement_frac_means', width=16, height=4)
    #
    # mu2_vs_mu3_ERsims_plot = (
    #         ggplot(output_agreement_frac_means_MELT_alltime, aes(x="mu_2", y="mu_3", colour="time", alpha=0.7)) +
    #         geom_point() +
    #         ggtitle("Prob. Any Node Pair Agrees (mu_2) vs. Prob. Neighbors Agree (mu_3)") +
    #         geom_abline(intercept=0, slope=1) +  # Adds a black line y=x to the plot
    #         guides(alpha=False))
    # mu2_vs_mu3_ERsims_plot.save('Sims_plot_mu2_vs_mu3', width=12, height=8)

    ###################################################################################
    ########################### Real Data section #####################################
    ###################################################################################
    # Run unit test for part (e)
    voter_funs.unit_test_mu_4()

    ###########################################
    #########  ** YOUR WORK HERE ** ###########
    ###########################################
    # Load the data into adj_matrix
    with open("datasets/adj_bool_mun25_n80.csv") as f:
        ncols = len(f.readline().split(','))
    A = np.genfromtxt("datasets/adj_bool_mun25_n80.csv", delimiter=',', skip_header=1, usecols=range(1, ncols + 1))

    Opinion1 = np.genfromtxt(".\datasets\opinions_mun25_elec1_n80.csv", dtype=int, delimiter=',', skip_header=1,
                             usecols=1, encoding="utf8")
    Opinion2 = np.genfromtxt(".\datasets\opinions_mun25_elec2_n80.csv", dtype=int, delimiter=',', skip_header=1,
                             usecols=1, encoding="utf8")

    #Plots with Opinion1
    nTrials = 200
    time_step = 64
    all_mu_1 = np.zeros((nTrials, time_step + 1), dtype=float)
    all_mu_2 = np.zeros((nTrials, time_step + 1), dtype=float)
    all_mu_3 = np.zeros((nTrials, time_step + 1), dtype=float)
    all_mu_4 = np.zeros((nTrials, time_step + 1), dtype=float)
    for tr in range(nTrials):
        ret_opinion = voter_mat(A, Opinion1, time_step)
        all_mu_1[tr] = [compute_opinion_fraction(ret_opinion[i]) for i in range(ret_opinion.shape[0])]
        all_mu_2[tr] = [compute_agreement_fraction(ret_opinion[i]) for i in range(ret_opinion.shape[0])]
        all_mu_3[tr] = [compute_neighbor_agreement_fraction(A, ret_opinion[i]) for i in range(ret_opinion.shape[0])]
        all_mu_4[tr] = [compute_euclid_norm_to_other_opinions(ret_opinion[i], Opinion2) for i in
                        range(ret_opinion.shape[0])]
        print(f'trial Number: {tr}')

    all_mu1_T = all_mu_1.transpose()
    all_mu1_TIndex = pd.DataFrame(all_mu1_T).assign(my_time=range(t + 1))
    mu1_long_form = pd.melt(all_mu1_TIndex, id_vars=['my_time'])
    mu1_long_form.rename(columns={"my_time": "time", "value": "mu_1", "variable": "trials"}, inplace=True)

    mu1_vs_time_plot = (
            ggplot(mu1_long_form, aes(x="time", y="mu_1", alpha=0.5)) +
            geom_line(aes(colour="trials")) +
            ggtitle("mu_1 vs time for Opinion 1") +
            guides(alpha=False) +
            geom_smooth())
    mu1_vs_time_plot.save('mu_1_vs_time_Opinion1', width=12, height=8)

    all_mu2_T = all_mu_2.transpose()
    all_mu2_TIndex = pd.DataFrame(all_mu2_T).assign(my_time=range(t + 1))
    mu2_long_form = pd.melt(all_mu2_TIndex, id_vars=['my_time'])
    mu2_long_form.rename(columns={"my_time": "time", "value": "mu_2", "variable": "trials"}, inplace=True)

    mu2_vs_time_plot = (
            ggplot(mu2_long_form, aes(x="time", y="mu_2", alpha=0.5)) +
            geom_line(aes(colour="trials")) +
            ggtitle("mu_2 vs time") +
            guides(alpha=False) +
            geom_smooth())
    mu2_vs_time_plot.save('mu_2_vs_time', width=12, height=8)

    all_mu3_T = all_mu_3.transpose()
    all_mu3_TIndex = pd.DataFrame(all_mu3_T).assign(my_time=range(t + 1))
    mu3_long_form = pd.melt(all_mu3_TIndex, id_vars=['my_time'])
    mu3_long_form.rename(columns={"my_time": "time", "value": "mu_3", "variable": "trials"}, inplace=True)

    mu3_vs_time_plot = (
            ggplot(mu3_long_form, aes(x="time", y="mu_3", alpha=0.5)) +
            geom_line(aes(colour="trials")) +
            ggtitle("mu_3 vs time") +
            guides(alpha=False) +
            geom_smooth())
    mu3_vs_time_plot.save('mu_3_vs_time', width=12, height=8)

    all_mu4_T = all_mu_4.transpose()
    all_mu4_TIndex = pd.DataFrame(all_mu4_T).assign(my_time=range(t + 1))
    mu4_long_form = pd.melt(all_mu4_TIndex, id_vars=['my_time'])
    mu4_long_form.rename(columns={"my_time": "time", "value": "mu_4", "variable": "trials"}, inplace=True)

    mu4_vs_time_plot = (
            ggplot(mu4_long_form, aes(x="time", y="mu_4", alpha=0.5)) +
            geom_line(aes(colour="trials")) +
            ggtitle("mu_4 vs time") +
            guides(alpha=False) +
            geom_smooth())
    mu4_vs_time_plot.save('mu_4_vs_time', width=12, height=8)


    all_mu_1_Op2 = np.zeros((nTrials, time_step + 1), dtype=float)
    all_mu_2_Op2 = np.zeros((nTrials, time_step + 1), dtype=float)
    all_mu_3_Op2 = np.zeros((nTrials, time_step + 1), dtype=float)
    for tr in range(nTrials):
        ret_opinion_2 = voter_mat(A, Opinion2, time_step)
        all_mu_1_Op2[tr] = [compute_opinion_fraction(ret_opinion_2[i]) for i in range(ret_opinion_2.shape[0])]
        all_mu_2_Op2[tr] = [compute_agreement_fraction(ret_opinion_2[i]) for i in range(ret_opinion_2.shape[0])]
        all_mu_3_Op2[tr] = [compute_neighbor_agreement_fraction(A, ret_opinion_2[i]) for i in range(ret_opinion_2.shape[0])]
        print(f'trial Number: {tr}')

    all_mu_1_Op2T = all_mu_1_Op2.transpose()
    all_mu_1_Op2TIndex = pd.DataFrame(all_mu_1_Op2T).assign(my_time=range(t + 1))
    mu1_long_form_Op2 = pd.melt(all_mu_1_Op2TIndex, id_vars=['my_time'])
    mu1_long_form_Op2.rename(columns={"my_time": "time", "value": "mu_1", "variable": "trials"}, inplace=True)

    mu1_vs_time_plot_Op2 = (
            ggplot(mu1_long_form_Op2, aes(x="time", y="mu_1", alpha=0.5)) +
            geom_line(aes(colour="trials")) +
            ggtitle("mu_1 vs time for Opinion 2") +
            guides(alpha=False) +
            geom_smooth())
    mu1_vs_time_plot_Op2.save('mu_1_vs_time_Opinion2', width=12, height=8)

    all_mu2_Op2T = all_mu_2_Op2.transpose()
    all_mu2_Op2TIndex = pd.DataFrame(all_mu2_Op2T).assign(my_time=range(t + 1))
    mu2_long_form_Op2 = pd.melt(all_mu2_Op2TIndex, id_vars=['my_time'])
    mu2_long_form_Op2.rename(columns={"my_time": "time", "value": "mu_2", "variable": "trials"}, inplace=True)

    mu2_vs_time_plot_Op2 = (
            ggplot(mu2_long_form_Op2, aes(x="time", y="mu_2", alpha=0.5)) +
            geom_line(aes(colour="trials")) +
            ggtitle("mu_2 vs time for Opinion2") +
            guides(alpha=False) +
            geom_smooth())
    mu2_vs_time_plot_Op2.save('mu_2_vs_time_Opinion2', width=12, height=8)

    all_mu3_Op2T = all_mu_3_Op2.transpose()
    all_mu3_Op2TIndex = pd.DataFrame(all_mu3_Op2T).assign(my_time=range(t + 1))
    mu3_long_form_Op2 = pd.melt(all_mu3_Op2TIndex, id_vars=['my_time'])
    mu3_long_form_Op2.rename(columns={"my_time": "time", "value": "mu_3", "variable": "trials"}, inplace=True)

    mu3_vs_time_plot_Op2 = (
            ggplot(mu3_long_form_Op2, aes(x="time", y="mu_3", alpha=0.5)) +
            geom_line(aes(colour="trials")) +
            ggtitle("mu_3 vs time for Opinion2") +
            guides(alpha=False) +
            geom_smooth())
    mu3_vs_time_plot_Op2.save('mu_3_vs_time_Opinion2', width=12, height=8)

    ####################################################################################
    ################ Influence Maximization on Municipality Network ####################
    ####################################################################################

    ###########################################
    #########  ** YOUR WORK HERE ** ###########
    ###########################################
    rows = A.shape[0]
    cols = A.shape[1]
    M = np.zeros((rows, cols), dtype=float)

    for i in range(0, rows):
        countN = Counter(A[i])
        prob = float((1. / countN[1]))
        M[i] = np.where(A[i] == 1, prob, 0.)

    tf = 21
    Mf = matrix_power(M, tf)
    rowF = np.array([1] * rows, dtype=float)
    col_sum = np.dot(rowF, Mf)
    p = np.argpartition(col_sum, -3)[-3:]

    print(f'indices: {p}')
    print(f'values: {col_sum[p]}')

    elec1_opinion = np.genfromtxt(".\datasets\opinions_mun25_elec1_n80.csv", dtype=int, delimiter=',', skip_header=1,
                             usecols=0, encoding="utf8")

    print(f'The three municipilaties with maximum influence: {elec1_opinion[p[0]]}, '
          f'{elec1_opinion[p[1]]}, {elec1_opinion[p[2]]}')
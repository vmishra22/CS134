import skeleton  # necessary because the unit testing functions herein test functions that are in YOUR script,
# so we need to import that script
import numpy as np
import random


#############################################
#############################################
#######   Functions provided for you  #######
#############################################
# !! NO NEED to change code in this file !! #
#############################################


# Compute square of Euclidean distance (norm) to vector that is stable (either all-0-vector or all-1-vector)
# Note that since all entries will be 0 or 1, squaring each element has no effect except dropping the '-' sign, (so we can skip to save on computation)
def compute_euclid_norm_to_EQB_squared(opinions):
    return ((np.float(1) / len(opinions)) * min(np.sum(np.abs(opinions - np.zeros(len(opinions)))),
                                                np.sum(np.abs(opinions - np.ones(len(opinions))))))


#############################################
# Helper function to initialize Erdos-Renyi random graphs
def make_symmetric_ER_w_all_self_loops(N, p):
    # Initialize an upper triangular random Boolean matrix, where p_{mn}(True) == p: 0<=p<=1 in the upper triangle and 0 in the lower triangle
    A = np.array([[False] * n + [random.random() < p] * (N - n) for n in range(N)])

    # Make A symmetric
    A = A + A.T

    # Make all self loops True
    np.fill_diagonal(A, True)
    return (A)


#############################################
# Helper function to run the algorithm over 2 vectors of values of length l1 and l2 and return the results in a l1 x l2 matrix (for heatmapping)
def run_voter_over_params(N, A_gen_fun, A_gen_fun_param_vec, opinions0_densities_vec, t, matrix_draws_per_A_param,
                          voter_runs_per_A_t):
    # Run counter
    run_counter = 0

    # Initialize 5-Dimensional arrays to hold all output summary statistics
    dims = (
        len(A_gen_fun_param_vec), matrix_draws_per_A_param, len(opinions0_densities_vec), voter_runs_per_A_t, (t + 1))
    number_of_runs = np.prod(dims) / (t + 1)
    output_opin_frac = np.zeros(dims)
    output_agreement_frac = np.zeros(dims)
    output_neighbor_agreement_frac = np.zeros(dims)
    # output_euclid_norm_to_EQB_squared = np.zeros(dims)
    # output_euclid_norm_to_opinions0 = np.zeros(dims)

    # Nested loop to end all nested loops. Who needs vectorization, anyways?
    for A_param_i in range(len(A_gen_fun_param_vec)):
        for A_draw_j in range(matrix_draws_per_A_param):

            # Generate an adjacency matrix using the A_gen_fun() method and argument A_param
            A = A_gen_fun(N, A_gen_fun_param_vec[A_param_i])

            for opinion_density_k in range(len(opinions0_densities_vec)):
                # Generate opinions vector with this density of people who have belief == True (rounded DOWN to nearest integer of people)
                opinions = np.array([True] * int(N * opinions0_densities_vec[opinion_density_k]) + \
                                    [False] * (N - int(N * opinions0_densities_vec[opinion_density_k])))

                for run_m in range(voter_runs_per_A_t):
                    run_counter = run_counter + 1
                    if (run_counter % 100) == 0:
                        print('Beginning run ' + str(run_counter) + ' of ' + str(number_of_runs))

                    # Do this parameter combination run_m times for this matrix A (so we can average the results and avoid focusing on an 'odd' run)
                    # opinions_out = voter(A, opinions, t_vec[t_l])
                    opinions_mat = skeleton.voter_mat(A, opinions, t)

                    # Summarize the results using our summary statistics and store these summary statistics
                    output_opin_frac[A_param_i, A_draw_j, opinion_density_k, run_m,] = \
                        [skeleton.compute_opinion_fraction(opinions_mat[i,]) for i in range(t + 1)]

                    output_agreement_frac[A_param_i, A_draw_j, opinion_density_k, run_m,] = \
                        [skeleton.compute_agreement_fraction(opinions_mat[i,]) for i in range(t + 1)]

                    output_neighbor_agreement_frac[A_param_i, A_draw_j, opinion_density_k, run_m,] = \
                        [skeleton.compute_neighbor_agreement_fraction(A, opinions_mat[i,]) for i in range(t + 1)]

                # output_euclid_norm_to_EQB_squared[A_param_i, A_draw_j, opinion_density_k, run_m, ] = \
                # 	[compute_euclid_norm_to_EQB_squared(opinions_mat[i,]) for i in range(t+1)]

                # output_euclid_norm_to_opinions0[A_param_i, A_draw_j, opinion_density_k, run_m, ] = \
                # 	[skeleton.compute_euclid_norm_to_other_opinions(opinions_mat[i,], opinions_mat[0,]) for i in range(t+1)]

    # Initialize holders averaging over all run_m and all A_draw_j (i.e. all re-runs of voter() with same params and all re-runs with same params but re-draws of random matrix A (with same p))
    output_opin_frac_means = np.nanmean(output_opin_frac, axis=(1, 3))
    output_agreement_frac_means = np.nanmean(output_agreement_frac, axis=(1, 3))
    output_neighbor_agreement_frac_means = np.nanmean(output_neighbor_agreement_frac, axis=(1, 3))
    # output_opin_frac_sd = np.std(output_opin_frac[i, :,k, l, ])
    # output_neighbor_agreement_frac_sd = np.std(output_neighbor_agreement_frac[i, :,k, l, ])
    # output_euclid_norm_to_EQB_squared_means = np.nanmean(output_euclid_norm_to_EQB_squared, axis=(1,3))
    # output_euclid_norm_to_opinions0_means = np.nanmean(output_euclid_norm_to_opinions0, axis=(1,3))

    return ([output_opin_frac,
             output_agreement_frac,
             output_neighbor_agreement_frac,
             output_opin_frac_means,
             output_agreement_frac_means,
             output_neighbor_agreement_frac_means,
             # output_euclid_norm_to_EQB_squared_means,
             # output_euclid_norm_to_opinions0_means
             ])


#############################################
def unit_test_voter_mat():
    # Inputs:
    N = 5

    # Initialize adjacency matrix
    A = np.zeros((N, N), dtype=bool)

    # Generate edges
    # A[0,1] = A[0,2] = A[0,4] = A[1,3] = A[1,4] = A[2,4] = True
    A[0, 2] = A[1, 2] = A[2, 3] = A[3, 4] = True

    # Make symmetric
    A = A + A.transpose()

    # Add self-loops
    np.fill_diagonal(A, True)

    ## Unit Test 1:
    ## With the following inputs, your algorithm should compute the final opinions vector as all True (or all 1)
    opinions_test1 = np.array([True] * N, dtype=bool)
    t_test1 = 3
    if np.array_equal(skeleton.voter_mat(A, opinions_test1, t_test1),
                      np.ones(((t_test1 + 1), len(opinions_test1)), dtype=bool)):
        print("voter_mat PASSED FIRST UNIT TEST")
    else:
        print("voter_mat FAILED FIRST UNIT TEST")

    ## Unit Test 2: Verifying each node's probability of opinion_i== "True"
    # opinions_test2 = np.array([False]*N, dtype=bool)
    # opinions_test2[[1,2,3]] = True
    opinions_test2 = np.array([True, True, False, True, False])
    t_test2 = 1
    skeleton.voter_mat(A, opinions_test2, t_test2)
    # print "SECOND UNIT TEST: **MANUALLY CHECK** that when opinions=="+str(opinions_test2)+", your vector of probabilities that opinion_i== True is [0.75, 0.75, 0.6666667, 0.5, 0.75]"
    print("voter_mat SECOND UNIT TEST: !*!*! MANUALLY CHECK !*!*! that when opinions==" + str(
        opinions_test2) + ", your vector of probabilities that opinion_i== True is [0.5, 0.5, 0.75, 0.3333333, 0.5] !*!*!!*!*!*!*!*!*!*!*!*!*!*!*!*!*!")


#############################################
def unit_test_mu1_mu2_mu3():
    # Inputs:
    N = 5

    # Set tolerance for rounding errors
    tolerance = 0.0001

    # Generate opinions
    opinions_testmus = np.array([True, True, False, True, False])

    # Test mu_1
    if ((abs(skeleton.compute_opinion_fraction(opinions_testmus) - 3.0 / 5) / (3.0 / 5)) > tolerance):
        print("compute_opinion_fraction FAILED unit test")
        print("with opinions: " + str(opinions_testmus) + " mu_1 should be 0.6")
    else:
        print("compute_opinion_fraction PASSED unit test")

    # Test mu_2
    if ((abs(skeleton.compute_agreement_fraction(opinions_testmus) - 0.4) / (0.4)) > tolerance):
        print("compute_agreement_fraction FAILED unit test")
        print("with opinions: " + str(opinions_testmus) + " mu_2 should be 0.4")
    else:
        print("compute_agreement_fraction PASSED unit test")

    # Test mu_3
    # Initialize adjacency matrix
    A = np.zeros((N, N), dtype=bool)

    # Generate edges
    A[0, 2] = A[1, 2] = A[2, 3] = A[3, 4] = True

    # Make symmetric
    A = A + A.transpose()

    # Add self-loops
    np.fill_diagonal(A, True)

    if skeleton.compute_neighbor_agreement_fraction(A, opinions_testmus) != 0.0:
        print("compute_neighbor_agreement_fraction FAILED its first unit test")
        print(f'With A=", {A}, " and opinions = ", str(opinions_testmus, " mu_3 should be 0')
    else:
        print("compute_neighbor_agreement_fraction PASSED its first unit test")

    opinions_testmu_3 = np.array([True, True, False, True, True])
    if ((abs(skeleton.compute_neighbor_agreement_fraction(A, opinions_testmu_3) - 1.0 / 4) / (1.0 / 4)) > tolerance):
        print("compute_neighbor_agreement_fraction FAILED its second unit test")
        print(f"With A= ", {str(A)}, " and opinions = " + str(opinions_testmu_3) + " mu_3 should be 0.25")
    else:
        print("compute_neighbor_agreement_fraction PASSED its second unit test")


#############################################
def unit_test_mu_4():
    opinions_mu4 = np.array([True, True, False, True, False])
    opinions2_mu4 = np.array([False, False, True, False, False])

    out = skeleton.compute_euclid_norm_to_other_opinions(opinions_mu4, opinions2_mu4)
    if ((abs(out - 4.0 / 5) / (4.0 / 5)) > 0.0001):
        print("compute_euclid_norm_to_other_opinions FAILED unit test")
        print("With opinions_mu4= " + str(opinions_mu4) + " and opinions2_mu4= " + str(
            opinions2_mu4) + " this should be 0.8")
    else:
        print("compute_euclid_norm_to_other_opinions PASSED unit test")

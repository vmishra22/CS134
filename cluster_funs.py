
from __future__ import division
import numpy as np
from math import sqrt
from scipy.cluster.vq import kmeans, kmeans2, whiten, vq
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import plot,show


#####################################################
##  Helper Functions (No need to change these)     ##
#####################################################


def procrustes(X, Y):
    """
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    This implementation is based on the Matlab procrustes() function.
    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centered Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    cities_lonlat = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(cities_lonlat, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    traceTcities_lonlat = s.sum()

    d = 1 + ssY/ssX - 2 * traceTcities_lonlat * normY / normX
    Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - np.dot(muY, T)

    # transformation values 
    tform = {'rotation':T, 'translation':c}

    return d, Z, tform




def cmdscale(dist_adjacency):
    """                                                                                       
    A function to perform Classic Multidimensional Scaling (based on Matlab's cmdscale function). 
    CMD scaling takes an adjacency matrix of pairwise distances between points and finds the best projection
    of these points in Euclidean space that recreates a 'map' of these points that best preserves their 
    pairwise distances (in up to (N-1) dimensions).                                            
                                                                                               
    INPUT:                                                                              
    D: the NxN symmetric distance matrix                                                            
                                                                                               
    OUTPUTs:                                                                                  
    Y: N X P np.array. Each of the P columns represents a Euclidean dimension (e.g. x, y, z, ...).
        Note that P <= N-1, and P will be equal to the number of positive eigenvalues.                                      
                                                                                               
    e : an N-long vector containing the eigenvalues                                                                                                                                                                 
    """

    # Get number of points                                                                        
    N = dist_adjacency.shape[0]
 
    # Generate the centering matrix to center the adjacency matrix of distances                                                                      
    H = np.eye(N) - np.ones((N, N))/N

    # Center our adjacency matrix of distances and get eigenvals, eigenvecs                                                                          
    eigenvalues, eigenvectors = np.linalg.eigh( -H.dot(dist_adjacency**2).dot(H)/2 )
 
    # Sort by eigenvalue in descending order                                                  
    idx_reversed   = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_reversed]
    eigenvectors = eigenvectors[:,idx_reversed]
 
    # Determine the Euclidean points coordinates using only the positive eigenvalues                 
    pos_eig_bool = eigenvalues > 0
    L = np.diag( np.sqrt(eigenvalues[pos_eig_bool]) )
    V = eigenvectors[:,pos_eig_bool]
    Y = V.dot(L)
 
    return Y, eigenvalues

















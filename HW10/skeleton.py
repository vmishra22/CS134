#from numpy import *
from __future__ import division
import cluster_funs ## IMPORTANT imports our helper functions
import numpy as np
import pandas as pd
from math import sqrt
from scipy.cluster.vq import kmeans, kmeans2, whiten, vq
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import plot,show
import scipy

## Mapping Tools
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon



def unit_test_ratio():
    A_test = pd.read_csv('A_for_ratio_unit_test.csv', delimiter=',').as_matrix()
    np.fill_diagonal(A_test, 0)
    clusters_test = pd.read_csv('clusts_for_ratio_unit_test.csv', delimiter=',').as_matrix()

    N = 1000
    #A_test = np.random.randint(low=1, high = N, size=N**2).reshape((N,N))
    #np.fill_diagonal(A_test, 0)
    #clusters_test = np.random.randint(1, 4, size=N)

    print(ratio(A_test, clusters_test))

    if (abs(ratio(A_test, clusters_test) - 0.995829043348) < 0.0001):
        print("Unit Test for Ratio passed" ); return
    else:
        print("Unit Test for Ratio failed. Answer should be near 1 in this unit test. Specifically, ratio should be 0.995829043348")




def ratio(A, clusters):
    return(0.0)






if __name__ == '__main__':
    ###############################################################
    #####################     Question 1     ######################
    ###############################################################

    ## YOUR WORK HERE ##
    ####################

    ## Part 1.C
    unit_test_ratio()

    #####################################################
    ##  EXAMPLE of COLORING MUNICIPALITIES OF COLOMBIA ##
    #####################################################

    # Generate the empty map of the area around Colombia
    map = Basemap(llcrnrlon=-65,llcrnrlat=-6,urcrnrlon=-80,urcrnrlat=15)

    #Load the shape files. 
    map.readshapefile('COL_adm2', name='municipalities', drawbounds=True)

    # The following checks the shapefile data and makes a list such that we can
    # reference each municipality by its index. Unfortunately, the shapefiles do NOT
    # have the actual municipality codes that everyone else uses -- they just have indices
    # from 1 to 1065. We have provided a csv file lookup table to convert back and forth
    # NOTE some lookups are missing. If you encounter one, then just leave that municipality uncolored 

    mun_idx = []
    for shape_dict in map.municipalities_info:
        # We will be using the column ID_2 to reference municipalities (see in the shapefile csv)
        mun_idx.append(shape_dict['ID_2'])

    # Read CSV to convert from shapefile ID's to municipality codes ("Cod.Municipio")
    codebook = pd.read_csv('COL_adm2_Conversion.csv', delimiter=',')
    
    # Convert to dictionary for quick lookups
    codebook_dict = dict(zip(list(codebook['Cod.Municipio']),list(codebook['ID_2'])))

    # Here's how we would color the municipality Leticia (ID_2==4; Cod.Municipio==91001) red:
    ax = plt.gca() # Get current plot axes
    seg = map.municipalities[mun_idx.index( codebook_dict[91001] )] ## THIS is the important part
    poly = Polygon(seg, facecolor='red',edgecolor='red')
    ax.add_patch(poly) # Add the colored in polygon to our map

    plt.show()

    









    ###############################################################
    #####################     Question 2     ######################
    ###############################################################

    ### PART 2.A ###
    ###############################################################

    # Load the list of latitude and longitude for each municipality
    mun_lonlat_DF = pd.read_csv('mun_latlon.csv', delimiter=',')

    # Cast to Numpy Matrix
    cities_lonlat = mun_lonlat_DF[['lon','lat']].as_matrix()

    # Plot the original map
    plt.figure(1, figsize=(10, 6), dpi=100)
    plt.scatter(cities_lonlat[:,0], cities_lonlat[:,1], c='b', s=30, marker='x', alpha=0.5)
    #plt.axes().set_aspect('equal', 'box')
    plt.title('Original Map of the Municipalities')
    plt.savefig("OriginalMap.png")
    plt.show()


    ## YOUR WORK HERE ##
    ####################




import matplotlib.pyplot as plt
import vtk
import numpy as np
import sys
import math
import os
import glob
from vtk.util.numpy_support import *
import pandas
from multiprocessing import Pool
from scipy.stats import kurtosis, skew
###################################################################

def read_vti(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# def get_jsdiv(dist1,dist2):
#     kldiv1=0
#     kldiv2=0
#     jsdiv=0
#     nbins = len(dist1)
#     for i in range(nbins):
#         if dist1[i] > 0.0 and dist2[i] > 0.0:
#             kldiv1 = kldiv1 + dist1[i]*np.log(dist1[i]/dist2[i])
            
#     for i in range(nbins):
#         if dist1[i] > 0.0 and dist2[i] > 0.0:
#             kldiv2 = kldiv2 + dist2[i]*np.log(dist2[i]/dist1[i])
            
#     jsdiv = 0.5*(kldiv1+kldiv2)        
#     return jsdiv 

def get_jsdiv(dist1,dist2):
    kldiv1=0
    kldiv2=0
    jsdiv=0
    nbins = len(dist1)

    avg_dist = (dist1 + dist2)/2.0

    for i in range(nbins):
        if dist1[i] > 0.0 and avg_dist[i] > 0.0:
            kldiv1 = kldiv1 + dist1[i]*np.log(dist1[i]/avg_dist[i])

    for i in range(nbins):
        if avg_dist[i] > 0.0 and dist2[i] > 0.0:
            kldiv2 = kldiv2 + dist2[i]*np.log(dist2[i]/avg_dist[i])  

    jsdiv = 0.5*(kldiv1+kldiv2)        
            
    return jsdiv               

def write_vti(filename,data):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()

def get_cumulative_hist(data, nbins):
    hist = np.histogram(data,nbins)[0]
    totPts = sum(hist)
    ## compute cumulative distribution
    hist = hist/float(totPts)
    cum_dist = np.cumsum(hist)
    return cum_dist

def get_emd(dist1,dist2):
    emd = 0
    for i in range(len(dist1)):
        emd = emd + np.absolute(dist1[i]-dist2[i])
    return emd


def bhattacharya_dist_gaussian(block_signature,feature):
    mean1 = block_signature[0]
    stdev1 = np.sqrt(block_signature[1])
    mean2 = feature[0]
    stdev2 = np.sqrt(feature[1])

    bhatta_dist = ((mean1-mean2)*(mean1-mean2))/(4*(stdev1+stdev2))

    if stdev1 > 0:
        num = 0.5*(stdev1+stdev2)
        denom = np.sqrt(stdev1*stdev2)
        val2 = 0.5*np.log(num/denom)
        bhatta_dist = bhatta_dist + val2

    return bhatta_dist


# def compare_moments(block_signature,feature):
#     diff = 0
#     weights = [0.25,0.25,0.25,0.25]
#     for i in range(len(feature)):
#         val = np.absolute(feature[i] - block_signature[i])
#         #val = np.power(val,0.75)
#         diff = diff + val*weights[i]
#     return diff

def compare_moments(block_signature,feature):
    diff = 0
    weights = [0.25,0.25]
    for i in range(2):
        val = np.absolute(feature[i] - block_signature[i])
        #val = np.power(val,0.75)
        diff = diff + val*weights[i]
    return diff    

def get_kldiv(dist1,dist2):
    kldiv=0
    nbins = len(dist1)
    for i in range(nbins):
        if dist1[i] > 0.0 and dist2[i] > 0.0:
            kldiv = kldiv + dist1[i]*np.log(dist1[i]/dist2[i])
    return kldiv

def statistical_comparison(inparam):

    ## parse params
    raw_data_file = inparam[0]
    feature_moments = inparam[1]
    tstep = inparam[2]
    feature_dist = inparam[3]
    nbins = inparam[4]
    mixing_ratio = inparam[5]
    
    ### call SLIC first: This calls a c++ executable
    slic_exec = 'slic'
    command = './' + slic_exec + ' ' + raw_data_file + ' ' + str(tstep)
    os.system(command)
    ####################################################

    cluster_file = '../out/slic_clusters/cluster_' +  str(tstep) + '.vti'
    varname1 = 'ClusterIds'
    varname2 = 'ImageScalars'
            
    cluster_data =  read_vti(cluster_file)
    raw_data = read_vti(raw_data_file)
    
    range_vals = cluster_data.GetPointData().GetArray(0).GetRange()
    num_clusters = int(range_vals[1]) + 1
    
    ## declare 2D list
    cluster_based_data=[[] for i in xrange(num_clusters)]

    np_cluster_array = vtk.util.numpy_support.vtk_to_numpy(cluster_data.GetPointData().GetArray(0))
    np_data_array = vtk.util.numpy_support.vtk_to_numpy(raw_data.GetPointData().GetArray(varname2))

    for i in range(len(np_cluster_array)):
        cluster_based_data[np_cluster_array[i]].append(np_data_array[i])

    cluster_moments_comp = []
    cluster_jsdiv = []
    mixed_vals = []

    ## Iterate over each cluster and compute metrics for comparison
    for i in range(num_clusters):

        if len(cluster_based_data[i]) == 0:
            print i, len(cluster_based_data[i])

        m1 = (np.average(cluster_based_data[i]))
        m2 = (np.var(cluster_based_data[i]))
        m3 = (skew(cluster_based_data[i]))
        m4 = (kurtosis(cluster_based_data[i]))
        cluster_moments = [m1,m2,m3,m4]
        #cluster_moments_comp.append(compare_moments(cluster_moments,feature_moments))
        cluster_moments_comp.append(bhattacharya_dist_gaussian(cluster_moments,feature_moments))

        cluster_histogram = np.histogram(cluster_based_data[i], nbins)[0]
        norm_cluster_histogram = cluster_histogram/float(len(cluster_based_data[i]))
        jsdiv = get_jsdiv(norm_cluster_histogram,feature_dist)
        cluster_jsdiv.append(jsdiv)

    ## Normalize the metric values  
    cluster_jsdiv = (cluster_jsdiv-np.min(cluster_jsdiv))/(np.max(cluster_jsdiv)-np.min(cluster_jsdiv))
    cluster_moments_comp = (cluster_moments_comp-np.min(cluster_moments_comp))/(np.max(cluster_moments_comp)-np.min(cluster_moments_comp))
    ## flip the values, now higher values are important to us    
    cluster_moments_comp = 1.0 - cluster_moments_comp
    cluster_jsdiv = 1.0 - cluster_jsdiv

    ## Do mixing of the two metric values
    for i in range(num_clusters):
        #val = cluster_jsdiv[i]*cluster_moments_comp[i]
        val = cluster_jsdiv[i]*mixing_ratio + (1-mixing_ratio)*cluster_moments_comp[i]
        mixed_vals.append(val)
    
    moments_arr = vtk.vtkDoubleArray()
    moments_arr.SetName('moment_val')
    jsdiv_arr = vtk.vtkDoubleArray()
    jsdiv_arr.SetName('jsdiv_val')
    mixed_arr = vtk.vtkDoubleArray()
    mixed_arr.SetName('mixed_val')

    for i in range(len(np_cluster_array)):
        cid =  cluster_data.GetPointData().GetArray(0).GetTuple1(i)
        moments_arr.InsertNextTuple1(cluster_moments_comp[int(cid)])
        jsdiv_arr.InsertNextTuple1(cluster_jsdiv[int(cid)])
        mixed_arr.InsertNextTuple1(mixed_vals[int(cid)])

    raw_data.GetPointData().AddArray(moments_arr) 
    raw_data.GetPointData().AddArray(jsdiv_arr)
    raw_data.GetPointData().AddArray(mixed_arr)
    
    outfile = '../out/mfix_local_grid/slic_compare_' + str(tstep) + '.vti'
    write_vti(outfile,raw_data)

##################################################################################################
nbins = 32
mixing_ratio = 0.2## 0.5

feature_dist_input = '../feature_dists/new_bubble_local_grid_1.csv' ##'../feature_dists/mfix_new_1.csv', mfix_bubble_datavals_3.csv
## load feature distribution
df = pandas.read_csv(feature_dist_input)
feature_data = np.asarray(df['ImageScalars'])

feature_dist = np.histogram(feature_data,nbins)[0]
feature_points = sum(feature_dist)
feature_dist = feature_dist/float(feature_points)

### compute moments
m1 = np.average(feature_data)
m2 = np.var(feature_data)
m3 = skew(feature_data)
m4 = kurtosis(feature_data)
feature_moments = [m1,m2,m3,m4] 

initstep = 408
finalstep = 409

# initstep = 470
# finalstep = 479

# ### multiprocess all timesteps
#data_path = '/disk1/MFIX_bubble_fields/'
#data_path = '/disk1/MFIX_active_matter_density_fields/'
data_path = '/disk1/MFIX_bubble_fields_localbound/'
inputfname = []

for file in sorted(os.listdir(data_path)):
    if file.endswith(".vti"):
        inputfname.append(os.path.join(data_path,file))

## sort by timestep numbers
inputfname.sort(key=lambda f: int(filter(str.isdigit, f)))

# Create a pool of worker processes, each able to use a CPU core
pool = Pool(processes=36)
args = [(inputfname[i-initstep], feature_moments, i, feature_dist, nbins, mixing_ratio) for i in range(initstep,finalstep)] 

## Execute the multiprocess code
pool.map(statistical_comparison, args)
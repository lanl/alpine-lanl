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

def write_vti(filename,data):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()

def compute_3d_to_1d_map(x,y,z,dimx,dimy,dimz):
    return x + dimx*(y+dimy*z)

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

def compare_moments(block_signature,feature):
    diff = 0
    weights = [0.25,0.25,0.25,0.25]
    for i in range(len(feature)):
        val = np.absolute(feature[i] - block_signature[i])
        #val = np.power(val,2.0)
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
    data_path = inparam[0]
    feature_moments = inparam[1]
    tstep = inparam[2]

    raw_data_file = data_path + str(tstep) + '.vti'
    
    ### call SLIC first: This calls a c++ executable
    slic_exec = 'slic'
    command = './' + slic_exec + ' ' + raw_data_file + ' ' + str(tstep)
    os.system(command)
    ####################################################

    cluster_file = '../out/slic_clusters/cluster_' +  str(tstep) + '.vti'

    varname1 = 'ClusterIds'
    varname2 = 'ImageScalars'
    nbins = 128
        
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

    for i in range(num_clusters):
        m1 = (np.average(cluster_based_data[i]))
        m2 = (np.var(cluster_based_data[i]))
        m3 = (skew(cluster_based_data[i]))
        m4 = (kurtosis(cluster_based_data[i]))
        cluster_moments = [m1,m2,m3,m4]
        
        cluster_moments_comp.append(compare_moments(cluster_moments,feature_moments))

    ## Normalize the metric values  
    cluster_moments_comp = (cluster_moments_comp-np.min(cluster_moments_comp))/(np.max(cluster_moments_comp)-np.min(cluster_moments_comp))
    ## flip the values, now higher values are important to us    
    cluster_moments_comp = 1.0 - cluster_moments_comp
    
    moments_arr = vtk.vtkDoubleArray()
    moments_arr.SetName('moment_val')

    for i in range(len(np_cluster_array)):
        cid =  cluster_data.GetPointData().GetArray(0).GetTuple1(i)
        moments_arr.InsertNextTuple1(cluster_moments_comp[int(cid)])

    raw_data.GetPointData().AddArray(moments_arr) 
    
    outfile = '../out/mfix_case_3/slic_moment_compare_' + str(tstep) + '.vti'
    write_vti(outfile,raw_data)

##################################################################################################

feature_dist_input = '../feature_dists/mfix_bubble_datavals_2.csv'
## load feature distribution
df = pandas.read_csv(feature_dist_input)
feature_data = np.asarray(df['ImageScalars'])

### compute moments
m1 = np.average(feature_data)
m2 = np.var(feature_data)
m3 = skew(feature_data)
m4 = kurtosis(feature_data)
feature_moments = [m1,m2,m3,m4]   

initstep = 75
finalstep = 85

# ### multiprocess all timesteps
data_path = '/disk1/MFIX_bubble_fields_highres/original_timestep_'

# Create a pool of worker processes, each able to use a CPU core
pool = Pool(processes=30)
args = [(data_path, feature_moments, i) for i in range(initstep,finalstep)] 

## Execute the multiprocess code
pool.map(statistical_comparison, args)
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
import random

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

### Compute 3D SSIM New
def compute_3D_SSIM_new(data1,data2, min_val, max_val):

    if len(data1) == 0:
        return 0
    else:
        mean1 = np.average(data1)
        mean2 = np.average(data2)
        var1 = np.var(data1)
        var2 = np.var(data2)
        #covar = np.abs(np.cov(data1,data2)[0][1])

        DR = max_val - min_val
        c1 = np.power((0.0001*DR),2)
        c2 = np.power((0.0003*DR),2)

        l_comp = (2*mean1*mean2 + c1)/(mean1*mean1 + mean2*mean2 + c1)
        c_comp = (2*var1*var2+c2)/(var1*var1 + var2*var2 + c2)

        if len(data1) != len(data2):
            random.shuffle(data2)
            data2 = data2[0:len(data1)]

        if np.min(data1) == 0 and np.max(data1) == 0:
            s_comp = 0.0
        elif np.min(data2) == 0 and np.max(data2) == 0:
            s_comp = 0.0
        else:   
            s_comp = np.abs(np.corrcoef(data1,data2)[0][1])

        ssim = l_comp*c_comp

        print l_comp, c_comp, s_comp, ssim

        return ssim    

def ssim_comparison(inparam):

    ## parse params
    data_path = inparam[0]
    feature_data = inparam[1]
    tstep = inparam[2]

    data_min = np.min(feature_data)
    data_max = np.max(feature_data)
     
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
    cluster_ssim_comp = []

    ## iterate over each cluster
    for i in range(num_clusters):
        ### compute SSIM value
        cluster_ssim_comp.append(compute_3D_SSIM_new(cluster_based_data[i],feature_data,data_min,data_max))

    ## Normalize the metric values  
    cluster_ssim_comp = (cluster_ssim_comp-np.min(cluster_ssim_comp))/(np.max(cluster_ssim_comp)-np.min(cluster_ssim_comp))
    ## flip the values, now higher values are important to us    
    #cluster_ssim_comp = 1.0 - cluster_ssim_comp
    
    ssim_arr = vtk.vtkDoubleArray()
    ssim_arr.SetName('ssim_val')

    for i in range(len(np_cluster_array)):
        cid =  cluster_data.GetPointData().GetArray(0).GetTuple1(i)
        ssim_arr.InsertNextTuple1(cluster_ssim_comp[int(cid)])

    raw_data.GetPointData().AddArray(ssim_arr) 
    
    outfile = '../out/mfix_case_3/slic_ssim_compare_' + str(tstep) + '.vti'
    write_vti(outfile,raw_data)

##################################################################################################

feature_data_input = '../feature_dists/mfix_bubble_datavals_2.csv'
## load feature distribution
df = pandas.read_csv(feature_data_input)
feature_data = np.asarray(df['ImageScalars'])

initstep = 75
finalstep = 85

# ### multiprocess all timesteps
data_path = '/disk1/MFIX_bubble_fields/original_timestep_'

# Create a pool of worker processes, each able to use a CPU core
pool = Pool(processes=5)
args = [(data_path, feature_data, i) for i in range(initstep,finalstep)] 

## Execute the multiprocess code
pool.map(ssim_comparison, args)
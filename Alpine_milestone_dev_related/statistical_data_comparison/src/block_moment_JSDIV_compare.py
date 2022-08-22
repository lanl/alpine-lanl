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
  
def get_jsdiv(dist1,dist2):
    kldiv1=0
    kldiv2=0
    jsdiv=0
    nbins = len(dist1)
    for i in range(nbins):
        if dist1[i] > 0.0 and dist2[i] > 0.0:
            kldiv1 = kldiv1 + dist1[i]*np.log(dist1[i]/dist2[i])
            
    for i in range(nbins):
        if dist1[i] > 0.0 and dist2[i] > 0.0:
            kldiv2 = kldiv2 + dist2[i]*np.log(dist2[i]/dist1[i])
            
    jsdiv = 0.5*(kldiv1+kldiv2)        
            
    return jsdiv

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

def statistical_comparison(inparam):

    ## parse params
    raw_data_file = inparam[0]
    feature_moments = inparam[1]
    tstep = inparam[2]
    feature_dist = inparam[3]
    nbins = inparam[4]
    mixing_ratio = inparam[5] 
    xbox = inparam[6]
    ybox = inparam[7]
    zbox = inparam[8]

    print 'processing tstep: ' + str(tstep)

    totPtsBlock = xbox*ybox*zbox   

    varname = 'ImageScalars'
    raw_data = read_vti(raw_data_file)
    dims = raw_data.GetDimensions()

    np_var_array = vtk.util.numpy_support.vtk_to_numpy(raw_data.GetPointData().GetArray(varname))
    np_var_array = np.reshape(np_var_array,(dims[2],dims[1],dims[0]))
    moment_array = np.zeros_like(np_var_array)
    jsdiv_array = np.zeros_like(np_var_array)
    combined_array = np.zeros_like(np_var_array)
    
    moment_comp_val = 0
    jsdiv_val = 0

    # Iterate over each block and compute SSIM
    for k in range(0,dims[2],zbox):
        for j in range(0,dims[1],ybox):
            for i in range(0,dims[0],xbox):
            
                ## note that np_var_array has x and z swapped. SO, k index in given first
                block_data = np_var_array[k:k+zbox, j:j+ybox, i:i+xbox] 
                block_data = block_data.reshape(totPtsBlock)

                m1 = (np.average(block_data))
                m2 = (np.var(block_data))
                m3 = (skew(block_data))
                m4 = (kurtosis(block_data))
                block_moments = [m1,m2,m3,m4]
                moment_comp_val = compare_moments(block_moments,feature_moments)
                moment_array[k:k+zbox, j:j+ybox, i:i+xbox] = moment_comp_val

                block_histogram = np.histogram(block_data, nbins)[0]
                norm_block_histogram = block_histogram/float(len(block_data))
                jsdiv_val = get_jsdiv(norm_block_histogram,feature_dist)
                jsdiv_array[k:k+zbox, j:j+ybox, i:i+xbox] = jsdiv_val

    ## Normalize the metric values  
    moment_array = (moment_array-np.min(moment_array))/(np.max(moment_array)-np.min(moment_array))
    jsdiv_array = (jsdiv_array-np.min(jsdiv_array))/(np.max(jsdiv_array)-np.min(jsdiv_array))
    ## flip the values, now higher values are important to us    
    jsdiv_array = 1.0 - jsdiv_array
    moment_array = 1.0 - moment_array
    
    for i in range(len(moment_array)):
        combined_array[i] = jsdiv_array[i]*mixing_ratio + (1-mixing_ratio)*moment_array[i]


    combined_array = combined_array.reshape(dims[0]*dims[1]*dims[2])
    combined_array_vtk = vtk.util.numpy_support.numpy_to_vtk(combined_array)
    combined_array_vtk.SetName('feature_similarity')   

    raw_data.GetPointData().AddArray(combined_array_vtk) 
    
    outfile = '../out/mfix_case_3/block_compare_moment_jsdiv_' + str(tstep) + '.vti'
    write_vti(outfile,raw_data)

##################################################################################################
xbox = 4
ybox = 4
zbox = 4
nbins = 64
mixing_ratio = 0.7

feature_dist_input = '../feature_dists/mfix_bubble_datavals_3.csv' ##'../feature_dists/mfix_new_1.csv'
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

initstep = 75
finalstep = 408

# ### multiprocess all timesteps
data_path = '/disk1/MFIX_bubble_fields/'
#data_path = '/disk1/MFIX_bubble_fields_highres/'
inputfname = []

for file in sorted(os.listdir(data_path)):
    if file.endswith(".vti"):
        inputfname.append(os.path.join(data_path,file))

## sort by timestep numbers
inputfname.sort(key=lambda f: int(filter(str.isdigit, f)))

# Create a pool of worker processes, each able to use a CPU core
pool = Pool(processes=30)
args = [(inputfname[i-initstep], feature_moments, i, feature_dist, nbins, mixing_ratio, xbox, ybox, zbox) for i in range(initstep,finalstep)] 

## Execute the multiprocess code
pool.map(statistical_comparison, args)
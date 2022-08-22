import vtk
import numpy as np
import sys
import math
import os
import glob
from vtk.util.numpy_support import *
import pandas
from multiprocessing import Pool
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

def write_vtu(filename,data):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()     

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

def statistical_comparison(inparam):

    ## parse params
    raw_data_file = inparam[0]
    feature_moments = inparam[1]
    tstep = inparam[2]
    outpath = inparam[3]
    confidence_th = inparam[4] 
    
    ### call SLIC first: This calls a c++ executable.
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
        cluster_moments = [m1,m2]
        cluster_moments_comp.append(bhattacharya_dist_gaussian(cluster_moments,feature_moments))
        
    ## Normalize the metric values  
    cluster_moments_comp = (cluster_moments_comp-np.min(cluster_moments_comp))/(np.max(cluster_moments_comp)-np.min(cluster_moments_comp))
    ## flip the values, now higher values are important to us    
    cluster_moments_comp = 1.0 - cluster_moments_comp
    
    moments_arr = vtk.vtkDoubleArray()
    moments_arr.SetName('feature_similarity')

    for i in range(len(np_cluster_array)):
        cid =  cluster_data.GetPointData().GetArray(0).GetTuple1(i)
        moments_arr.InsertNextTuple1(cluster_moments_comp[int(cid)])
        
    raw_data.GetPointData().AddArray(moments_arr) 
    
    #outfile = '../out/mfix_local_grid/slic_compare_' + str(tstep) + '.vti'
    outfile = outpath + '/slic_compare_' + str(tstep) + '.vti'
    write_vti(outfile,raw_data)

    ##############################################
    ##Do thresholding and connected components
    raw_data.GetPointData().SetActiveScalars('feature_similarity')
    thresholding = vtk.vtkThreshold()
    thresholding.ThresholdByUpper( confidence_th )
    thresholding.SetInputData(raw_data)
    thresholding.Update()
    #outfile = outpath +  '/slic_threshold_' + str(tstep) + '.vtu'
    #write_vtu(outfile,thresholding.GetOutput())

    seg = vtk.vtkConnectivityFilter()
    seg.SetInputConnection(thresholding.GetOutputPort())
    seg.SetExtractionModeToAllRegions()
    seg.ColorRegionsOn()
    seg.Update()
    outfile = outpath +  '/slic_segmented_' + str(tstep) + '.vtu'
    write_vtu(outfile,seg.GetOutput())

##################################################################################################
feature_dist_input = '../feature_dists/new_bubble_local_grid_1.csv'
## load feature distribution
df = pandas.read_csv(feature_dist_input)
feature_data = np.asarray(df['ImageScalars'])

### compute moments
m1 = np.average(feature_data)
m2 = np.var(feature_data)
feature_moments = [m1,m2] 
print m1, m2


################################################# 
## TODO: Set Params and Datapaths here
initstep = 75
finalstep = 76
confidence_th = 0.9 
# ### multiprocess all timesteps
data_path = '/home/sdutta/Desktop/' ## '/disk1/MFIX_bubble_fields_localbound/'
outpath = '/home/sdutta/Desktop/' ## '../out/mfix_local_grid'

inputfname = []
for file in sorted(os.listdir(data_path)):
    if file.endswith(".vti"):
        inputfname.append(os.path.join(data_path,file))

## sort by timestep numbers
inputfname.sort(key=lambda f: int(filter(str.isdigit, f)))

# Create a pool of worker processes, each able to use a CPU core
pool = Pool(processes=24)
args = [(inputfname[i-initstep], feature_moments, i, outpath, confidence_th) for i in range(initstep,finalstep)] 

## Execute the multiprocess code
pool.map(statistical_comparison, args)
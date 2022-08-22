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

### Compute 3D SSIM
def compute_3D_SSIM(data1,data2, min_val, max_val):
    if len(data1) != len(data2):
        data2 = data2[0:len(data1)]

    mean1 = np.average(data1)
    mean2 = np.average(data2)
    var1 = np.var(data1)
    var2 = np.var(data2)
    covar = np.abs(np.cov(data1,data2)[0][1])
    
    DR = max_val - min_val
    
    c1 = np.power((0.01*DR),2)
    c2 = np.power((0.03*DR),2)
    ssim = ((2*mean1*mean2 + c1)*(2*covar+c2))/((mean1*mean1 + mean2*mean2 + c1)*(var1*var1+var2*var2+c2))
    return ssim

### Compute 3D SSIM New
def compute_3D_SSIM_new(data1,data2, min_val, max_val):

    # if len(data1) != len(data2):
    #     data2 = data2[0:len(data1)]

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

    if np.min(data1) == 0 and np.max(data1) == 0:
    	s_comp = 0.0
    elif np.min(data2) == 0 and np.max(data2) == 0:
    	s_comp = 0.0
    else:	
    	if len(data1) != len(data2):
            random.shuffle(data2)
            data2 = data2[0:len(data1)]
    	
    	s_comp = np.abs(np.corrcoef(data1,data2)[0][1])

    ssim = l_comp*c_comp*s_comp

    return ssim    

def gen_ssim_field_single_tstep(inparam):
    
    ## parse params
    tstep_num = inparam[0]
    out_file_path1 = inparam[1]
    varname = inparam[2]
    final_feature_data = inparam[3]
    xbox = inparam[4]
    ybox = inparam[5]
    zbox = inparam[6]
    totPtsBlock = inparam[7]
    data_min = inparam[8]
    data_max = inparam[9]
 
    ##load data
    data_file = data_path + str(tstep_num) + '.vti'
    data = read_vti(data_file)
    np_var_array = vtk.util.numpy_support.vtk_to_numpy(data.GetPointData().GetArray(varname))
    np_var_array = np.reshape(np_var_array,(zdim,ydim,xdim))
    classified_array = np.zeros_like(np_var_array)
    dims = data.GetDimensions()
    
    # Iterate over each block and compute SSIM
    for k in range(0,dims[2],zbox):
        for j in range(0,dims[1],ybox):
            for i in range(0,dims[0],xbox):
            
                ## note that np_var_array has x and z swapped. SO, k index in given first
                block_data = np_var_array[k:k+zbox, j:j+ybox, i:i+xbox] 
                block_data = block_data.reshape(totPtsBlock)
                
                ssim_val = compute_3D_SSIM_new(block_data,final_feature_data, data_min, data_max)
                classified_array[k:k+zbox, j:j+ybox, i:i+xbox] = ssim_val
                
    classified_array = classified_array.reshape(xdim*ydim*zdim)
    classified_array_vtk = vtk.util.numpy_support.numpy_to_vtk(classified_array)
    classified_array_vtk.SetName('feature_similarity')
    data.GetPointData().AddArray(classified_array_vtk)

    out_fname = out_file_path1 + str(tstep_num) + '.vti'
    write_vti(out_fname,data)

## MFIX bubble data
#########################################################################
xdim = 256
ydim = 64
zdim = 256
initstep = 75
tsteps = 85
feature_tstep = 250
feature_data_file = '/disk1/MFIX_bubble_fields_highres/original_timestep_' + str(feature_tstep) + '.vti'

## boundary feature 1: tstep 250
# xmin = 32
# xmax = 40
# ymin = 0
# ymax = 8
# zmin = 202
# zmax = 210

#  boundary feature 2: tstep 250
xmin = 46
xmax = 54
ymin = 0
ymax = 8
zmin = 46
zmax = 50

# ##  boundary feature 3: tstep 190
# xmin = 13
# xmax = 17
# ymin = 0
# ymax = 8
# zmin = 220
# zmax = 224

# ##  bubble feature 1: tstep 175
# xmin = 84
# xmax = 88
# ymin = 0
# ymax = 8
# zmin = 170
# zmax = 174

data_path = '/disk1/MFIX_bubble_fields_highres/original_timestep_'
varname = 'ImageScalars'
out_file_path1 = '../out/mfix_case_3/block_ssim_field_'

#############################################################################################

xbox = xmax - xmin
ybox = ymax - ymin
zbox = zmax - zmin

xbox = 4
ybox = 4
zbox = 4

totPtsBlock = xbox*ybox*zbox

## Get the block data which will be treated as the feature of interest
feature_selection_field = read_vti(feature_data_file)
feature_data = vtk.util.numpy_support.vtk_to_numpy(feature_selection_field.GetPointData().GetArray(varname))

data_max = np.max(feature_data)
data_min = np.min(feature_data)

feature_data = np.reshape(feature_data,(zdim,ydim,xdim))
feature_data = feature_data[zmin:zmax, ymin:ymax, xmin:xmax]
shape = np.shape(feature_data)
final_feature_data = np.reshape(feature_data,shape[0]*shape[1]*shape[2])

# Create a pool of worker processes, each able to use a CPU core
pool = Pool(processes=10)
args = [(i,out_file_path1, varname, final_feature_data, xbox, ybox, zbox, totPtsBlock, data_min, data_max) \
		for i in range(initstep,tsteps)] 
## Execute the multiprocess code		
xx = pool.map(gen_ssim_field_single_tstep, args)
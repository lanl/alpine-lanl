import sys
import subprocess
import re
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cmath
import random
import matplotlib.cm as cm
import glob
import vtk
from vtk.util.numpy_support import *
import pandas as pd
from multiprocessing import Pool
from vtk.util import numpy_support
import pymp


from mi_functions import *
#######################################################################


# # ## load vti data: MFIX fcc insitu data testing
#################################################################
# data_path = '/Users/sdutta/Desktop/sim_fields_density/'
# outpath = '../output/time_varying_mfix/'
# startT = 50900
# endT = 51480 #51410
# window = 10
# numBins=128
# fname = 'density_'
# endfname = ''
# varname = 'density' #'feature_similarity' #'ImageScalars'
# density_th = 12
# feature_size_th = 1500

# # ## load vti data
#################################################################
# data_path = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_local_grid/'
# outpath = '../output/time_varying_mfix/'
# startT = 350
# endT = 360
# window = 1
# numBins=128
# fname = 'slic_compare_'
# endfname = ''
# varname = 'ImageScalars' #'feature_similarity' #'ImageScalars'
# density_th = 10
# dataset = 'mfix'


# # ## load vti data highres
#################################################################
# data_path = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_density_fields_highres/'
# outpath = '../output/time_varying_mfix/'
# startT = 12010
# endT = 12500
# window = 10
# numBins=128
# #fname = 'insitu_simfield_'
# fname = 'fcc_highres_raw'
# varname = 'ImageScalars' #'similarity'
# density_th = 80


# # ## Vortex data set
# ################################################################
# data_path = '/Users/sdutta/Data/vortex_vti/'
# outpath = '../output/time_varying_vortex/'
# startT = 1
# endT = 5
# window = 1
# numBins=128
# fname = 'vortex_'
# varname = 'ImageScalars'
# density_th = 7
# dataset = 'vortex'


## Isabel data set
################################################################
data_path = '/Users/sdutta/Data/Isabel_Pressure_timevarying/'
outpath = '../output/time_varying_isabel/'
startT = 1
endT = 20
window = 1
numBins=128
fname = 'Pf'
endfname = '.binLE.raw_corrected_2_subsampled'
varname = 'ImageScalars'
density_th = -500
dataset = 'isabel'


# # ## Tornado moving
# data_path = '/Users/sdutta/Data/Tornado_moving/'
# outpath = '../output/time_varying_tornado/'
# startT = 1
# endT = 50 # 50
# window = 1
# numBins=128
# fname = 'tornado_lambda2_'
# endfname = '.vti'
# varname = 'ImageScalars'
# density_th = -0.003
# dataset = 'tornado'


# # ## 3d_cylinder data set
# ################################################################
# data_path = '/Users/sdutta/Data/3d_cylinder/'
# outpath = '../output/time_varying_cylinder/'
# startT = 1
# endT = 50
# window = 1
# numBins=128
# fname = 'cylinder_lambda2_'
# varname = 'ImageScalars'
# density_th = -0.005
# dataset = 'cylinder'


# # ## Asteroid data set
# ################################################################
# data_path = '/Users/sdutta/Data/asteroid_300X300X300/'
# outpath = '../output/time_varying_asteroid/'
# startT = 0
# endT = 50
# window = 1
# numBins=128
# fname = 'pv_insitu_300x300x300_'
# varname = 'v03'
# density_th = 0.5
# dataset = 'asteroid'

##################################################################################################
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

####################################################

asteroid_tsteps = []
if dataset == 'asteroid':
    list_of_files = sorted( filter( os.path.isfile, glob.glob(data_path + '*') ) )
    for i in range(len(list_of_files)):
        file = list_of_files[i].split('_')[4]
        file = file.split('.')[0]
        asteroid_tsteps.append(int(file))

    asteroid_tsteps = sorted(asteroid_tsteps)
#####################################################

out_field_fused = vtk.vtkImageData()

I1Arr = vtk.vtkFloatArray()
I1Arr.SetName("I1_fused")
I1TimeArr = vtk.vtkFloatArray()
I1TimeArr.SetName("I1_Time")
densityFusedArr = vtk.vtkFloatArray()
densityFusedArr.SetName("data_fused")

## clean up previous results
#os.system('rm ' + outpath + '*.vti')
os.system('rm ' + outpath + 'out_*.vti')

for tstep in range(startT,endT,window):

    load_start = time.time()

    if tstep%10 == 0:
        print ('processing tstep: ' + str(tstep))
  
    if dataset == 'mfix' or dataset == 'vortex':
        # for mfix and vortex
        file1 = data_path + fname + str(tstep)  + '.vti'
        file2 = data_path + fname + str(tstep+window) + '.vti'

    elif dataset == 'isabel':
        ## for Isabel
        if tstep < 10:
            ttstep = '0' + str(tstep)
        else:
            ttstep = str(tstep)
        file1 = data_path + fname + ttstep + endfname +'.vti'
        if tstep+window < 10:
            ttstep = '0' + str(tstep+window)
        else:
            ttstep = str(tstep+window)
        file2 = data_path + fname + ttstep + endfname +'.vti'

    elif dataset == 'tornado':
        file1 = data_path + fname + str(tstep)  + '.vti'
        file2 = data_path + fname + str(tstep+window) + '.vti'

    elif dataset == 'cylinder':
        file1 = data_path + fname + str(tstep)  + '.vti'
        file2 = data_path + fname + str(tstep+window) + '.vti'

    elif dataset == 'asteroid':
        file1 = data_path + fname + str(asteroid_tsteps[tstep])  + '.vti'
        file2 = data_path + fname + str(asteroid_tsteps[tstep+window]) + '.vti'

    #################################################################
    arrayName1 = varname
    arrayName2 = varname

    R1 = vtk.vtkXMLImageDataReader()
    R1.SetFileName(file1)
    R1.Update()
    R2 = vtk.vtkXMLImageDataReader()
    R2.SetFileName(file2)
    R2.Update()

    dataArray1 = R1.GetOutput().GetPointData().GetArray(arrayName1)
    dataArray2 = R2.GetOutput().GetPointData().GetArray(arrayName2)
    dims = R1.GetOutput().GetDimensions()

    out_field_fused.SetDimensions(R1.GetOutput().GetDimensions())
    out_field_fused.SetSpacing(R1.GetOutput().GetSpacing())

    var1 = numpy_support.vtk_to_numpy(dataArray1)
    var2 = numpy_support.vtk_to_numpy(dataArray2)

    load_end = time.time()

    compute_start = time.time()

    ## compute histogram
    numSamples = np.shape(var1)[0]
    Array1 = np.histogram(var1,bins=numBins)[0]
    Array2 = np.histogram(var2,bins=numBins)[0]
    ArrayComb = np.histogram2d(var1,var2,bins=numBins)[0]

    ## compute SMI
    I11,I12 = compute_I1(Array1,Array2,ArrayComb,numSamples,numBins)

    #I12 = compute_pointwise_mutual_information(Array1,Array2,ArrayComb,numSamples,numBins)

    compute_end = time.time()

    fusion_start = time.time()

    ## Handle first time step
    if tstep == startT:
        minval=np.min(var2)
        maxval=np.max(var2)
        for i in range(len(var2)):
            binid = int(((var2[i] - minval)/(maxval-minval))*(numBins-1))

            if dataset == 'mfix' or dataset == 'isabel' or dataset == 'tornado' or dataset == 'cylinder':
                if var2[i] < density_th:
                    I1Arr.InsertNextTuple1(I12[binid])
                    I1TimeArr.InsertNextTuple1(tstep)
                    densityFusedArr.InsertNextTuple1(var2[i])
                else:
                    I1Arr.InsertNextTuple1(I12[binid])
                    I1TimeArr.InsertNextTuple1(tstep-1)
                    densityFusedArr.InsertNextTuple1(var2[i])

            if dataset == 'vortex' or dataset == 'asteroid':
                if var2[i] > density_th:
                    I1Arr.InsertNextTuple1(I12[binid])
                    I1TimeArr.InsertNextTuple1(tstep)
                    densityFusedArr.InsertNextTuple1(var2[i])
                else:
                    I1Arr.InsertNextTuple1(I12[binid])
                    I1TimeArr.InsertNextTuple1(tstep-1)
                    densityFusedArr.InsertNextTuple1(var2[i])
 
    else:
        minval=np.min(var2)
        maxval=np.max(var2)
        for i in range(len(var2)):
            binid = int(((var2[i] - minval)/(maxval-minval))*(numBins-1))
            curr_I1_val = I12[binid]

            if dataset == 'mfix' or dataset == 'isabel' or dataset == 'tornado' or dataset == 'cylinder':
                if var2[i] < density_th and curr_I1_val > I1Arr.GetTuple1(i):
                    I1Arr.SetTuple1(i,curr_I1_val)
                    I1TimeArr.SetTuple1(i,tstep)
                    densityFusedArr.SetTuple1(i,var2[i])

            if dataset == 'vortex' or dataset == 'asteroid':
                if var2[i] > density_th and curr_I1_val > I1Arr.GetTuple1(i):
                    I1Arr.SetTuple1(i,curr_I1_val)
                    I1TimeArr.SetTuple1(i,tstep)
                    densityFusedArr.SetTuple1(i,var2[i])

    fusion_end = time.time()

    partial_start = time.time()

    out_field_time = vtk.vtkImageData()
    out_field_time.SetDimensions(R1.GetOutput().GetDimensions())
    out_field_time.SetSpacing(R1.GetOutput().GetSpacing())

    ## generate partial fields
    #out_field_time.GetPointData().AddArray(I1Arr)
    out_field_time.GetPointData().AddArray(I1TimeArr)
    out_field_time.GetPointData().AddArray(densityFusedArr)
    #out_field_time.GetPointData().AddArray(dataArray2)
    ## write partial fused field at each tstep
    fname_out = outpath + 'out_step_' + str(tstep) + '.vti'
    write_vti(fname_out, out_field_time)

    partial_end = time.time()


    print (tstep, load_end-load_start, 
                    compute_end-compute_start, 
                    fusion_end-fusion_start, 
                    partial_end-partial_start)


##################################################################################################
## add the fused fields to final output data
#out_field_fused.GetPointData().AddArray(I1Arr)
out_field_fused.GetPointData().AddArray(I1TimeArr)
out_field_fused.GetPointData().AddArray(densityFusedArr)
#out_field_fused.GetPointData().AddArray(dataArray2)

### Write the fused fields out
out_fname = outpath + 'fused_I1.vti'
write_vti(out_fname, out_field_fused)
    
      



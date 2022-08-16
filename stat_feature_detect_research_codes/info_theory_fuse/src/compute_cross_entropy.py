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


# ## Isabel data set
# ################################################################
# data_path = '/Users/sdutta/Data/Isabel_Pressure_timevarying/'
# outpath = '../output/time_varying_isabel/'
# startT = 1
# endT = 20
# window = 1
# numBins=128
# fname = 'Pf'
# endfname = '.binLE.raw_corrected_2_subsampled'
# varname = 'ImageScalars'
# density_th = -500
# dataset = 'isabel'


# ## Tornado moving
data_path = '/Users/sdutta/Data/Tornado_moving/'
outpath = '../output/time_varying_tornado/'
startT = 1
endT = 25
window = 1
numBins=128
fname = 'tornado_lambda2_'
endfname = '.vti'
varname = 'ImageScalars'
density_th = -0.003
dataset = 'tornado'

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
    PMI = compute_pointwise_mutual_information(Array1,Array2,ArrayComb,numSamples,numBins)

    compute_end = time.time()

    fusion_start = time.time()

    ## Handle first time step
    if tstep == startT:

 
    else:

       
    fusion_end = time.time()






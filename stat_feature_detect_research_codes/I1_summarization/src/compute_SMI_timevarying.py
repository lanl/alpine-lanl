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
import vtk

from mi_functions import *
#######################################################################

# # ## load vti data
# data_path = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_local_grid/'
# outpath = '../output/time_varying_mfix/'
# startT = 75
# endT = 407
# window = 1
# numBins=64
# fname = 'slic_compare_'
# varname = 'ImageScalars'


# # ## load vti data highres
# data_path = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_simfields_highres/'
# outpath = '../output/time_varying_mfix/'
# startT = 12000
# endT = 13000
# window = 10
# numBins=256
# fname = 'insitu_simfield_'
# varname = 'similarity'


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
# density_th = -0.005
# dataset = 'tornado'

## Isabel data set
################################################################
data_path = '/Users/sdutta/Data/Isabel_Pressure_timevarying/'
outpath = '../output/isabel_bin_study/256bins/'
startT = 20
endT = 30
window = 1
numBins=256
fname = 'Pf'
endfname = '.binLE.raw_corrected_2_subsampled'
varname = 'ImageScalars'
density_th = -500
dataset = 'isabel'

for tstep in range(startT,endT,window):

    if tstep%5 == 0:
        print ('processing tstep: ' + str(tstep))
    
    # file1 = data_path + fname + str(tstep) + '.vti'
    # file2 = data_path + fname + str(tstep+window) + '.vti'

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

    arrayName1 = varname
    arrayName2 = varname
    outfile = outpath + 'SMI_field_' + str(tstep) + '_' + str(tstep+window) + '.vti'

    R1 = vtk.vtkXMLImageDataReader()
    R1.SetFileName(file1)
    R1.Update()
    dataArray1 = R1.GetOutput().GetPointData().GetArray(arrayName1)

    R2 = vtk.vtkXMLImageDataReader()
    R2.SetFileName(file2)
    R2.Update()
    dataArray2 = R2.GetOutput().GetPointData().GetArray(arrayName2)

    var1=np.zeros(dataArray1.GetNumberOfTuples()) 
    var2=np.zeros(dataArray1.GetNumberOfTuples()) 

    for i in range(dataArray1.GetNumberOfTuples()):
        var1[i] = dataArray1.GetTuple1(i)
        var2[i] = dataArray2.GetTuple1(i)

    dims = R1.GetOutput().GetDimensions()

    ## compute histogram
    numSamples = np.shape(var1)[0]
    Array1 = np.histogram(var1,bins=numBins)[0]
    Array2 = np.histogram(var2,bins=numBins)[0]
    ArrayComb = np.histogram2d(var1,var2,bins=numBins)[0]

    ## compute SMI
    I11,I12,I21,I22,I31,I32 = compute_specific_mutual_information(Array1,Array2,ArrayComb,numSamples,numBins)

    ## write SMI and PMI volumes

    ###################################################
    ## for var1
    I11Arr = vtk.vtkFloatArray()
    I11Arr.SetName("I11")
    I21Arr = vtk.vtkFloatArray()
    I21Arr.SetName("I21")
    I31Arr = vtk.vtkFloatArray()
    I31Arr.SetName("I31")

    minval=np.min(var1)
    maxval=np.max(var1)
    for i in range(len(var1)):
        binid = int(((var1[i] - minval)/(maxval-minval))*(numBins-1))
        I11Arr.InsertNextTuple1(I11[binid])
        I21Arr.InsertNextTuple1(I21[binid])
        I31Arr.InsertNextTuple1(I31[binid])

    ###################################################
    ## for var2
    I12Arr = vtk.vtkFloatArray()
    I12Arr.SetName("I12")
    I22Arr = vtk.vtkFloatArray()
    I22Arr.SetName("I22")
    I32Arr = vtk.vtkFloatArray()
    I32Arr.SetName("I32")

    minval=np.min(var2)
    maxval=np.max(var2)
    for i in range(len(var2)):
        binid = int(((var2[i] - minval)/(maxval-minval))*(numBins-1))
        I12Arr.InsertNextTuple1(I12[binid])
        I22Arr.InsertNextTuple1(I22[binid])
        I32Arr.InsertNextTuple1(I32[binid])

    ####################################################
    ### write PMI volume
    pmi_field = vtk.vtkImageData()
    pmi_field.SetDimensions(R1.GetOutput().GetDimensions())
    pmi_field.SetSpacing(R1.GetOutput().GetSpacing())

    #pmi_field.GetPointData().AddArray(PMIArr)     
    pmi_field.GetPointData().AddArray(I12Arr)     
    pmi_field.GetPointData().AddArray(I22Arr)    
    pmi_field.GetPointData().AddArray(I32Arr)
    pmi_field.GetPointData().AddArray(I11Arr)     
    pmi_field.GetPointData().AddArray(I21Arr)    
    pmi_field.GetPointData().AddArray(I31Arr)

    W3 = vtk.vtkXMLImageDataWriter()
    W3.SetInputData(pmi_field)
    W3.SetFileName(outfile)
    W3.Write()  



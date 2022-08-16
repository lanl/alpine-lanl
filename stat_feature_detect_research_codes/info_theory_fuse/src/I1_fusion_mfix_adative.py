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
import glob
import vtk
from vtk.util.numpy_support import *
import pandas as pd
from multiprocessing import Pool
from vtk.util import numpy_support

from mi_functions import *
#######################################################################


# # ## load vti data: MFIX fcc data for all time steps to evaluate offline timings
# #################################################################################################
# data_path = '/Users/sdutta/Codes/IEEE_Bigdata_MFIX_insitu_analysis/fcc_density_fields_all_raw/'
# outpath = '../output/mfix_insitu_raw/'
# startT = 25000
# endT = 31000
# window = 10
# numBins=128
# fname = 'fcc_raw'
# endfname = ''
# varname = 'ImageScalars'
# density_th = 12
# min_window = 50
# feature_size_th = 750
# num_pts = 128*16*128


# ## load vti data: MFIX highres fcc data for all time steps to evaluate offline timings
#################################################################################################
data_path = '/Users/sdutta/Codes/IEEE_Bigdata_MFIX_insitu_analysis/fcc_highres_density_fields_all_raw/'
outpath = '../output/mfix_highres_insitu_raw/'
startT = 15000
endT = 17500
window = 10
numBins=128
fname = 'fcc_raw'
endfname = ''
varname = 'ImageScalars'
density_th = 100
min_window = 50
feature_size_th = 500
num_pts = 128*16*128


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

#####################################################
IO_time = 0
compute_time = 0

I1Arr = vtk.vtkFloatArray()
I1Arr.SetName("I1_fused")
I1Arr.SetNumberOfTuples(num_pts)
I1TimeArr = vtk.vtkFloatArray()
I1TimeArr.SetName("I1_Time")
I1TimeArr.SetNumberOfTuples(num_pts)
densityFusedArr = vtk.vtkFloatArray()
densityFusedArr.SetName("data_fused")
densityFusedArr.SetNumberOfTuples(num_pts)

prev_bubble_count = 0
prev_time_step_stored = startT

for tstep in range(startT,endT,window):

    io_start = time.time()

    file1 = data_path + fname + str(tstep)  + '.vti'
    file2 = data_path + fname + str(tstep+window) + '.vti'

    #################################################################
    arrayName1 = varname
    arrayName2 = varname

    R1 = vtk.vtkXMLImageDataReader()
    R1.SetFileName(file1)
    R1.Update()
    R2 = vtk.vtkXMLImageDataReader()
    R2.SetFileName(file2)
    R2.Update()

    io_end = time.time()
    IO_time = IO_time + io_end - io_start

    compute_start = time.time()

    dataArray1 = R1.GetOutput().GetPointData().GetArray(arrayName1)
    dataArray2 = R2.GetOutput().GetPointData().GetArray(arrayName2)
    dims = R1.GetOutput().GetDimensions()
    var1 = numpy_support.vtk_to_numpy(dataArray1)
    var2 = numpy_support.vtk_to_numpy(dataArray2)

    ## estimate feature number at current time
    current_data = R2.GetOutput()
    current_data.GetPointData().SetActiveScalars(varname)
    thresholding = vtk.vtkThreshold()
    thresholding.ThresholdByLower(density_th)
    thresholding.SetInputData(current_data)
    seg = vtk.vtkConnectivityFilter()
    seg.SetInputConnection(thresholding.GetOutputPort())
    seg.SetExtractionModeToAllRegions()
    seg.ColorRegionsOn()
    seg.Update()

    ## compute histogram
    numSamples = np.shape(var1)[0]
    Array1 = np.histogram(var1,bins=numBins)[0]
    Array2 = np.histogram(var2,bins=numBins)[0]
    ArrayComb = np.histogram2d(var1,var2,bins=numBins)[0]
    ## compute SMI
    I11,I12 = compute_I1(Array1,Array2,ArrayComb,numSamples,numBins)

    ## Handle first time step
    if tstep == startT:
        minval=np.min(var2)
        maxval=np.max(var2)
        for i in range(len(var2)):
            binid = int(((var2[i] - minval)/(maxval-minval))*(numBins-1))

            if var2[i] < density_th:
                I1Arr.SetTuple1(i,I12[binid])
                I1TimeArr.SetTuple1(i,tstep+window)
                densityFusedArr.SetTuple1(i,var2[i])
            else:
                I1Arr.SetTuple1(i,I12[binid])
                I1TimeArr.SetTuple1(i,tstep)
                densityFusedArr.SetTuple1(i,var2[i])

        curr_number_of_features = 0
        for jj in range (seg.GetNumberOfExtractedRegions()):
            thresholding1 = vtk.vtkThreshold()
            thresholding1.ThresholdBetween(jj,jj)
            thresholding1.SetInputData(seg.GetOutput())
            thresholding1.Update()
            obj = thresholding1.GetOutput()
            if obj.GetNumberOfPoints() > feature_size_th:
              curr_number_of_features = curr_number_of_features+1;
        
        print('first time step:' + str(startT))

        prev_bubble_count = curr_number_of_features

    elif tstep > startT:

        curr_number_of_features = 0
        for jj in range (seg.GetNumberOfExtractedRegions()):
            thresholding1 = vtk.vtkThreshold()
            thresholding1.ThresholdBetween(jj,jj)
            thresholding1.SetInputData(seg.GetOutput())
            thresholding1.Update()
            obj = thresholding1.GetOutput()
            if obj.GetNumberOfPoints() > feature_size_th:
              curr_number_of_features = curr_number_of_features+1;

        ## Fusion condition
        if curr_number_of_features == prev_bubble_count:

            print ('normal fusion: ' + str(tstep+window))
            minval=np.min(var2)
            maxval=np.max(var2)
            for i in range(len(var2)):
                binid = int(((var2[i] - minval)/(maxval-minval))*(numBins-1))
                curr_I1_val = I12[binid]
                
                if var2[i] < density_th and curr_I1_val > I1Arr.GetTuple1(i):
                    I1Arr.SetTuple1(i,curr_I1_val)
                    I1TimeArr.SetTuple1(i,tstep+window)
                    densityFusedArr.SetTuple1(i,var2[i])

            prev_bubble_count = curr_number_of_features


        ## Still fuse
        elif (curr_number_of_features != prev_bubble_count) and ((tstep + window - prev_time_step_stored) <= min_window):
            print ('forced fusion: ' + str(tstep+window))
            minval=np.min(var2)
            maxval=np.max(var2)
            for i in range(len(var2)):
                binid = int(((var2[i] - minval)/(maxval-minval))*(numBins-1))
                curr_I1_val = I12[binid]
                
                if var2[i] < density_th and curr_I1_val > I1Arr.GetTuple1(i):
                    I1Arr.SetTuple1(i,curr_I1_val)
                    I1TimeArr.SetTuple1(i,tstep+window)
                    densityFusedArr.SetTuple1(i,var2[i])

            prev_bubble_count = curr_number_of_features


        else:
            print ('dumping data and reinitialization: ' + str(tstep+window))
            
            ## dump the current fused data
            out_field_fused = vtk.vtkImageData()
            out_field_fused.SetDimensions(R1.GetOutput().GetDimensions())
            out_field_fused.SetSpacing(R1.GetOutput().GetSpacing())
            out_field_fused.GetPointData().AddArray(I1TimeArr)
            out_field_fused.GetPointData().AddArray(densityFusedArr)
            out_field_fused.GetPointData().AddArray(dataArray2)
            ### Write the fused fields out
            out_fname = outpath + 'out_fused_' + str(tstep+window) + '.vti'
            write_vti(out_fname, out_field_fused)

            ## reinitialize the arrays
            minval=np.min(var2)
            maxval=np.max(var2)
            for i in range(len(var2)):
                binid = int(((var2[i] - minval)/(maxval-minval))*(numBins-1))
                curr_I1_val = I12[binid]
                I1Arr.SetTuple1(i,curr_I1_val)
                I1TimeArr.SetTuple1(i,tstep+window)
                densityFusedArr.SetTuple1(i,var2[i])

            prev_time_step_stored = tstep+window
            prev_bubble_count = curr_number_of_features

    compute_end = time.time()
    compute_time = compute_time + compute_end - compute_start

#################################################################
print (compute_time,IO_time)
outfile = outpath + 'timings_total.txt'
np.savetxt(outfile,[compute_time,IO_time],delimiter=',')


    
      



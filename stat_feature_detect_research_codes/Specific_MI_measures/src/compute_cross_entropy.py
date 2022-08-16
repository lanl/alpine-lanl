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
from vtk.util.numpy_support import *
import pandas as pd
from multiprocessing import Pool
from vtk.util import numpy_support

from mi_functions import *
#######################################################################



# ## load vti data: MFIX fcc insitu data testing
data_path = '/Users/sdutta/Desktop/sim_fields_density/'
outpath = '../output/time_varying_mfix/'
startT = 50900
endT = 51480 #51410
window = 10
numBins=128
fname = 'density_'
endfname = ''
varname = 'density' #'feature_similarity' #'ImageScalars'
density_th = 12
feature_size_th = 1500

# # ## load vti data
# data_path = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_local_grid/'
# outpath = '../output/time_varying_mfix/'
# startT = 290
# endT = 300
# window = 1
# numBins=128
# fname = 'slic_compare_'
# endfname = ''
# varname = 'ImageScalars' #'feature_similarity' #'ImageScalars'
# density_th = 10


# # ## load vti data highres
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
# data_path = '/Users/sdutta/Data/vortex_vti/'
# outpath = '../output/time_varying_vortex/'
# startT = 1
# endT = 10
# window = 1
# numBins=128
# fname = 'vortex_'
# varname = 'ImageScalars'
# density_th = 7


# # ## Isabel data set
# data_path = '/Users/sdutta/Data/Isabel_Pressure_timevarying/'
# outpath = '../output/time_varying_isabel/'
# startT = 30
# endT = 40
# window = 1
# numBins=128
# fname = 'Pf'
# endfname = '.binLE.raw_corrected_2_subsampled'
# varname = 'ImageScalars'
# density_th = -500

##################################################################################################


def write_vtu(filename,data):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()

out_field_fused = vtk.vtkImageData()
out_field_time = vtk.vtkImageData()

I1Arr = vtk.vtkFloatArray()
I1Arr.SetName("I1_fused")
I1TimeArr = vtk.vtkFloatArray()
I1TimeArr.SetName("I1_Time")

for tstep in range(startT,endT,window):

    # if tstep%50 == 0:
    #     print ('processing tstep: ' + str(tstep))
    
    # for mfix and vortex
    file1 = data_path + fname + str(tstep)  + '.vti'
    file2 = data_path + fname + str(tstep+window) + '.vti'

    # ## for Isabel
    # if tstep < 10:
    #     ttstep = '0' + str(tstep)
    # else:
    #     ttstep = str(tstep)
    # file1 = data_path + fname + ttstep + endfname +'.vti'

    # if tstep+window < 10:
    #     ttstep = '0' + str(tstep+window)
    # else:
    #     ttstep = str(tstep+window)
    # file2 = data_path + fname + ttstep + endfname +'.vti'


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

    ## compute histogram
    numSamples = np.shape(var1)[0]
    Array1 = np.histogram(var1,bins=numBins)[0]
    Array2 = np.histogram(var2,bins=numBins)[0]
    ArrayComb = np.histogram2d(var1,var2,bins=numBins)[0]

    ## compute SMI
    I11,I12,I21,I22,I31,I32 = compute_specific_mutual_information(Array1,Array2,ArrayComb,numSamples,numBins)


    ## Handle first time step
    if tstep == startT:
        minval=np.min(var2)
        maxval=np.max(var2)
        for i in range(len(var2)):
            binid = int(((var2[i] - minval)/(maxval-minval))*(numBins-1))
            I1Arr.InsertNextTuple1(I12[binid])
            I1TimeArr.InsertNextTuple1(tstep)
 
    else:
        minval=np.min(var2)
        maxval=np.max(var2)
        for i in range(len(var2)):
            binid = int(((var2[i] - minval)/(maxval-minval))*(numBins-1))
            curr_I1_val = I12[binid]

            if var2[i] < density_th and curr_I1_val > I1Arr.GetTuple1(i): #MFIX and Isabel
            #if var2[i] > density_th and curr_I1_val > I1Arr.GetTuple1(i):  #VORTEX
                I1Arr.SetTuple1(i,curr_I1_val)
                I1TimeArr.SetTuple1(i,tstep)

    R1.GetOutput().GetPointData().SetActiveScalars('density') 

    thresholding = vtk.vtkThreshold()
    thresholding.ThresholdByLower( density_th )
    thresholding.SetInputData(R1.GetOutput())
    thresholding.Update()
    seg = vtk.vtkConnectivityFilter()
    seg.SetInputConnection(thresholding.GetOutputPort())
    seg.SetExtractionModeToAllRegions()
    seg.ColorRegionsOn()
    seg.Update()
    
    ug = seg.GetOutput()
    numseg = seg.GetNumberOfExtractedRegions()
    #write_vtu(outpath + "/out.vtu",ug)

    count=0

    new_grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints() 
    for jj in range(numseg):
        thresholding1 = vtk.vtkThreshold()
        thresholding1.ThresholdBetween( jj,jj )
        thresholding1.SetInputData(ug)
        thresholding1.Update()
        obj = thresholding1.GetOutput()
               
              
        if obj.GetNumberOfPoints() > feature_size_th:
            count=count+1
            for kk in range(obj.GetNumberOfPoints()):
                points.InsertNextPoint(obj.GetPoint(kk))
                #print(obj.GetPoint(kk))

    
    new_grid.SetPoints(points)
    fnameout = outpath + 'out_filtered_' + str(tstep) + '.vtu'
    write_vtu(fnameout,new_grid)




    print ('num segments at time step: ' + str(tstep) + ' is: ' + str(numseg) + ' and after filtering is: ' + str(count))

out_field_fused.GetPointData().AddArray(I1Arr)
out_field_fused.GetPointData().AddArray(I1TimeArr)


################################################################################################
### Write the fused fields out
W = vtk.vtkXMLImageDataWriter()
W.SetInputData(out_field_fused)
out_fname = outpath + 'out_field_fused.vti'
W.SetFileName(out_fname)
W.Write()
    
      



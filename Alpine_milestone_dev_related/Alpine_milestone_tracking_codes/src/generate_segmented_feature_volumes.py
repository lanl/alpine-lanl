## Post-hoc feature tracking algorithm development.
## Algorithm developed for ECP Alpine Project.
## This code is delivered as part of Alpine 16-14 P6 activity.
## Please contact Soumya Dutta (sdutta@lanl.gov) for any questions.
## Unauthorized use of this code is prohibited.


## This script takes input as a set of similairty fields from a range of time steps.
## Then the script generates segmented feature volumes for each time step.
## Each time step results multiple volumes (vti files), one per feature.
## later these features are used for visualization

#############################################

import matplotlib.pyplot as plt
import vtk
import numpy as np
import sys
import math
import os
import glob
from vtk.util.numpy_support import *
from multiprocessing import Pool
from vtk.util import numpy_support
import pickle
import pandas as pd

#############################################

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
    
def extract_and_dump_feature_vti(single_feature,data):
    
    dims = data.GetDimensions()
    gbounds = data.GetBounds()
    numPts = dims[0]*dims[1]*dims[2]
    
    feature_field = vtk.vtkImageData()
    feature_field.SetDimensions(dims)
    feature_field.SetExtent(data.GetExtent())
    feature_field.SetSpacing(data.GetSpacing())
    feature_field.SetOrigin(data.GetOrigin())
    
    field = vtk.vtkDoubleArray()
    field.SetNumberOfTuples(numPts)
    field.SetName('feature_similarity')
    
    for i in range(numPts):
        field.SetTuple1(i,0.3)
   
    numFeaturePts = single_feature.GetNumberOfPoints()
    
    for i in range(numFeaturePts):
        pts = single_feature.GetPoint(i)
        xval = int(((pts[0] - gbounds[0])/(gbounds[1]-gbounds[0]))*dims[0])
        yval = int(((pts[1] - gbounds[2])/(gbounds[3]-gbounds[2]))*dims[1])
        zval = int(((pts[2] - gbounds[4])/(gbounds[5]-gbounds[4]))*dims[2])
        
        if xval > dims[0]-1:
            xval = dims[0]-1
        
        if yval > dims[1]-1:
            yval = dims[1]-1
            
        if zval > dims[2]-1:
            zval = dims[2]-1    
            
        index = compute_3d_to_1d_map(xval,yval,zval,dims[0],dims[1],dims[2])
        
        if index > dims[0]*dims[1]*dims[2] or index < 0:
            print (xval,yval,zval)
        
        val = data.GetPointData().GetArray('feature_similarity').GetTuple1(index)
        #val = data.GetPointData().GetArray('similarity').GetTuple1(index)
        field.SetTuple1(index,val)
        
        if xval > dims[0] or yval >dims[1] or zval > dims[2]:
            print (xval,yval,zval)
            
    field.SetTuple1(0,0.0)       
    feature_field.GetPointData().AddArray(field)
        
    return feature_field
    
def segment_feature(fname,confidence_th,size_threshold, tstep, outpath):
    
    data = read_vti(fname)
    data.GetPointData().SetActiveScalars('feature_similarity')    
    gbounds = data.GetBounds()

    thresholding = vtk.vtkThreshold()
    thresholding.ThresholdByUpper( confidence_th )
    thresholding.SetInputData(data)
    seg = vtk.vtkConnectivityFilter()
    seg.SetInputConnection(thresholding.GetOutputPort())
    seg.SetExtractionModeToLargestRegion()
    seg.Update()

    segmentation = vtk.vtkConnectivityFilter()
    segmentation.SetInputConnection(thresholding.GetOutputPort())
    segmentation.SetExtractionModeToAllRegions()
    segmentation.ColorRegionsOn()
    segmentation.Update()

    ug = segmentation.GetOutput()
    num_segments = segmentation.GetNumberOfExtractedRegions()
    
    ## compute volumes of each bubble:
    bubble_volumes = np.zeros(num_segments)
    for i in range(ug.GetPointData().GetArray('RegionId').GetNumberOfTuples()):
        regionId = int(ug.GetPointData().GetArray('RegionId').GetTuple(i)[0])
        bubble_volumes[regionId] = bubble_volumes[regionId]+1     
    
    idx=0
    find_max_topmost_xvals=[]
    for i in range(num_segments):
        if  bubble_volumes[i] > size_threshold:
            thresholding2 = vtk.vtkThreshold()
            thresholding2.SetInputData(ug)
            thresholding2.ThresholdBetween(i,i)
            thresholding2.Update()
            single_feature = thresholding2.GetOutput()
            feature_pts = numpy_support.vtk_to_numpy(single_feature.GetPoints().GetData())
            max_x = np.max(feature_pts[:,0])
            find_max_topmost_xvals.append([max_x,i,idx])
            idx=idx+1
            
    ## identify the indices of features which are actually topmost and not desired bubbles
    find_max_topmost_xvals = np.asarray(find_max_topmost_xvals)
    top_indices=[]
    for i in range(len(find_max_topmost_xvals)):
        diff = np.abs(gbounds[1]-find_max_topmost_xvals[i,0])
        if diff < 0.00005:
            top_indices.append(int(find_max_topmost_xvals[i,1]))      
    
    ## filter out the top most undesired features
    for i in range(num_segments):
        if  bubble_volumes[i] > size_threshold and i not in(top_indices):
            thresholding2 = vtk.vtkThreshold()
            thresholding2.SetInputData(ug)
            thresholding2.ThresholdBetween(i,i)
            thresholding2.Update()
            single_feature = thresholding2.GetOutput()
            feature_field = extract_and_dump_feature_vti(single_feature,data)
            outfname = outpath + 'segmented_feature_' + str(i) + '_' + str(tstep) + '.vti'
            write_vti(outfname,feature_field)


#################################################################################
## Parameters
###############
init_time = 75
end_time = 100
## Folder that contains all the feature similarity fields (ideally generated in situ)
input_path = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_local_grid/'
## Folder that will contain the segmented feature volumes (vti files)
outpath = '../out/segmented_volumes/'
## similarity threshold for segmentation
confidence_th = 0.92
## feature size threshold. The value indicates number of voxels. Any feature below this threshold will be discarded.
size_threshold = 100

#################################################################################

## Run for all timesteps
for ii in range(init_time,end_time):
	print ('Processing time step: ' + str(ii))
	inpfname = input_path + 'slic_compare_' + str(ii) + '.vti'
	segment_feature(inpfname,confidence_th,size_threshold, ii, outpath)

print ('done processing all time steps')


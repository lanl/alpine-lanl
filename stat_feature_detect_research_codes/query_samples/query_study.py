import numpy as np
import random
import platform
import sys
import os
import math
import time
import matplotlib
from matplotlib import pyplot as plt
import vtk

## for isabel
raw_data_file = 'data/isabel_p_25.vti'
sampled_file = 'data/isabel_p_25_hist_grad_rand_pymp_0.05.vtp'
outfile1 = 'output/recon_query_isabel.vtp'
outfile2 = 'output/recon_query_isabel.vti'
outfile3 = 'output/raw_query_isabel.vtp'
var_name = 'ImageFile'
var_name1 = 'ImageFile'
query_th1 = -500
query_th2 = -5000

# ## for combustion
# raw_data_file = 'data/jet_mixfrac_0041.dat_2_subsampled.vti'
# sampled_file = 'data/jet_mixfrac_0041.dat_2_subsampled_hist_grad_rand_pymp_0.05.vtp'
# outfile1 = 'output/recon_query_mixfrac.vtp'
# outfile2 = 'output/recon_query_mixfrac.vti'
# outfile3 = 'output/raw_query_mixfrac.vtp'
# var_name = 'ImageFile'
# var_name1 = 'ImageFile'
# query_th1 = 0.3
# query_th2 = 0.5

##################################################################################################

def compute_3d_to_1d_map(x,y,z,dimx,dimy,dimz):
    index = x + dimx*(y+dimy*z)
    return index

def getDist(pts1,pts2):
    dist = (pts1[0]-pts2[0])*(pts1[0]-pts2[0]) + (pts1[1]-pts2[1])*(pts1[1]-pts2[1]) + (pts1[2]-pts2[2])*(pts1[2]-pts2[2])
    return dist

## load data
#reader = vtk.vtkGenericDataObjectReader()
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(raw_data_file)
reader.Update()
raw_data = reader.GetOutput()

reader1 = vtk.vtkXMLPolyDataReader()
reader1.SetFileName(sampled_file)
reader1.Update()
sampled_pts = reader1.GetOutput()

extent = raw_data.GetExtent()
spacing = raw_data.GetSpacing()
origin = raw_data.GetOrigin()
dims = raw_data.GetDimensions()

## query raw data
query_pts_raw = vtk.vtkPoints()
query_pts_physical_raw = vtk.vtkPoints()
query_arr_raw = vtk.vtkDoubleArray()
query_arr_raw.SetName(var_name)

for i in range(dims[0]*dims[1]*dims[2]):
    vo2 = raw_data.GetPointData().GetArray(var_name).GetTuple1(i)
    #if vo2 >= query_th1 and vo2 <= query_th2: ## for others
    if vo2 <= query_th1 and vo2 >= query_th2:   ## for isabel 
        pts = raw_data.GetPoint(i)        
        query_arr_raw.InsertNextTuple1(vo2)   
        query_pts_physical_raw.InsertNextPoint(pts)
        
print ('Raw points selected: ' + str(query_pts_physical_raw.GetNumberOfPoints()))  

pdata_raw = vtk.vtkPolyData()
pdata_raw.SetPoints(query_pts_physical_raw)
pdata_raw.GetPointData().AddArray(query_arr_raw)
writer1 = vtk.vtkXMLPolyDataWriter()
writer1.SetInputData(pdata_raw)
writer1.SetFileName(outfile3)
writer1.Write()

## Query sampled data
###################################################################

## create new image data
new_data = vtk.vtkImageData()
new_data.SetDimensions(dims)
new_data.SetSpacing(spacing)
new_data.SetOrigin(origin)

## create point arrays
query_pts = vtk.vtkPoints()
query_pts_physical = vtk.vtkPoints()
query_arr = vtk.vtkDoubleArray()
query_arr.SetName(var_name)

query_arr_new = vtk.vtkDoubleArray()
query_arr_new.SetName(var_name)
query_arr_new.SetNumberOfTuples(dims[0]*dims[1]*dims[2])

query_arr_new1 = vtk.vtkDoubleArray()
query_arr_new1.SetName(var_name1)
query_arr_new1.SetNumberOfTuples(dims[0]*dims[1]*dims[2])

for i in range(dims[0]*dims[1]*dims[2]):
    query_arr_new1.SetTuple1(i,0.0)
    query_arr_new.SetTuple1(i,0.0)

for i in range(sampled_pts.GetNumberOfPoints()):
    vo2 = sampled_pts.GetPointData().GetArray(var_name1).GetTuple1(i)
    #if vo2 >= query_th1 and vo2 <= query_th2: ## for others
    if vo2 <= query_th1 and vo2 >= query_th2: ## for isabel     
        pts = sampled_pts.GetPoint(i)        
        ii = int((pts[0] - origin[0])/spacing[0] + 0.5)
        jj = int((pts[1] - origin[1])/spacing[1] + 0.5)
        kk = int((pts[2] - origin[2])/spacing[2] + 0.5)
        query_pts.InsertNextPoint([ii,jj,kk])
        query_arr.InsertNextTuple1(vo2)   
        query_pts_physical.InsertNextPoint(pts)
        
        index = compute_3d_to_1d_map(ii,jj,kk,dims[0],dims[1],dims[2])
        query_arr_new.SetTuple1(index,vo2)
        
        query_arr_new1.SetTuple1(index,vo2)
        
print ('Sampled points selected: ' + str(query_pts.GetNumberOfPoints())) 

pdata = vtk.vtkPolyData()
pdata.SetPoints(query_pts_physical)
pdata.GetPointData().AddArray(query_arr)

writer1 = vtk.vtkXMLPolyDataWriter()
writer1.SetInputData(pdata)
writer1.SetFileName(outfile1)
writer1.Write()


### splat values to neighbors
################################################################################
window = 2

for p  in range(query_pts.GetNumberOfPoints()):

    pts = query_pts.GetPoint(p)
    vo2 = query_arr_new.GetTuple1(p)
    
    i=int(pts[0])
    j=int(pts[1])
    k=int(pts[2])
    
    for kk in range(k-window,k+window):
        for jj in range(j-window,j+window):
            for ii in range(i-window,i+window):
                
                ## check if in bound
                if ii >= 0 and ii < dims[0] and jj >= 0 and jj < dims[1] and kk >= 0 and kk <dims[2]:
                    
                    ## ignore the central point
                    if kk == k and ii== i and jj == j: 
                        continue
                    else:
                        index1 = compute_3d_to_1d_map(ii,jj,kk,dims[0],dims[1],dims[2])

                        if query_arr_new.GetTuple1(index1) == 0: 
                            v2 = query_arr_new1.GetTuple1(index1)
                            pts1 = (ii,jj,kk)
                            dist = getDist(pts,pts1)
                            
                            v2 = v2 + vo2/(dist)
                            query_arr_new1.SetTuple1(index1,v2)


div_fac = np.power(window*2,3)
for i in range(query_arr_new1.GetNumberOfTuples()):
    v = query_arr_new1.GetTuple1(i)
    v1 = query_arr_new.GetTuple1(i)
    if v1 == 0:    
        query_arr_new1.SetTuple1(i,v/div_fac)
        
new_data.GetPointData().AddArray(query_arr_new1)                            

## write splat volume out
writer = vtk.vtkXMLImageDataWriter()
writer.SetInputData(new_data)
writer.SetFileName(outfile2)
writer.Write()

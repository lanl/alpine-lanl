import vtk
import numpy as np
import sys
import math
import os
import glob
from vtk.util.numpy_support import *
from vtk.util import numpy_support
import pandas
from multiprocessing import Pool
import re
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
    
def write_multiblock(filename,data):
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update() 

def write_vtp(filename,data):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()    
    
def read_plt(filename):
    reader = vtk.vtkAMReXParticlesReader()
    reader.SetPlotFileName(filename)
    reader.Update()
    return reader.GetOutput()

def compute_3d_to_1d_map(x,y,z,dimx,dimy,dimz):

    return x + dimx*(y+dimy*z)

def filter_particles(particle_data,sim_field,var_name,confidence_th):

	numPieces = particle_data.GetBlock(0).GetNumberOfPieces()
	#print("total pieces:", numPieces)

	## compute global bound
	xmin = []
	xmax = []
	ymin = []
	ymax = []
	zmin = []
	zmax = []
	for i in range(numPieces):
	    piece_bound = particle_data.GetBlock(0).GetPiece(i).GetBounds()
	    xmin.append(piece_bound[0])
	    xmax.append(piece_bound[1])
	    ymin.append(piece_bound[2])
	    ymax.append(piece_bound[3])
	    zmin.append(piece_bound[4])
	    zmax.append(piece_bound[5])

	spatial_range_x = [np.min(xmin), np.max(xmax)]
	spatial_range_y = [np.min(ymin), np.max(ymax)]
	spatial_range_z = [np.min(zmin), np.max(zmax)]

	#print (spatial_range_x, spatial_range_y, spatial_range_z)


	## Combine all the data points to one array
	NBlocks = particle_data.GetNumberOfBlocks()
	#print("total blocks:",NBlocks)

	mpData = particle_data.GetBlock(0)
	numPieces = mpData.GetNumberOfPieces()
	#print("total pieces:", numPieces)

	classified_array = sim_field.GetPointData().GetArray(var_name)

	filtered_particles = vtk.vtkPolyData()
	filtered_pts_arr = vtk.vtkPoints()

	for i in range(numPieces):
	    data = mpData.GetPiece(i)
	    
	    if data is not None:
	        allpts = data.GetPoints()

	        for i in range(data.GetNumberOfPoints()):
	            pts = allpts.GetPoint(i)

	            xBinId = int((pts[0]-spatial_range_x[0])/(spatial_range_x[1]-spatial_range_x[0])*nbins[0])
	            if xBinId==nbins[0]:
	                xBinId=xBinId-1
	            yBinId = int((pts[1]-spatial_range_y[0])/(spatial_range_y[1]-spatial_range_y[0])*nbins[1])
	            if yBinId==nbins[1]:
	                yBinId=yBinId-1
	            zBinId = int((pts[2]-spatial_range_z[0])/(spatial_range_z[1]-spatial_range_z[0])*nbins[2])
	            if zBinId==nbins[2]:
	                zBinId=zBinId-1

	            ## Get one-D ID
	            oneD_idx = compute_3d_to_1d_map(xBinId,yBinId,zBinId,nbins[0],nbins[1],nbins[2])

	            feature_value = classified_array.GetTuple1(oneD_idx)

	            if feature_value > confidence_th:
	                filtered_pts_arr.InsertNextPoint(pts)

	filtered_particles.SetPoints(filtered_pts_arr)

	return filtered_particles    

def extract_and_dump_feature_vti(single_feature,data):
    
    dims = data.GetDimensions()
    gbounds = data.GetBounds()
    numPts = dims[0]*dims[1]*dims[2]
    
    feature_field = vtk.vtkImageData()
    feature_field.SetDimensions(dims)
    feature_field.SetExtent(data.GetExtent())
    feature_field.SetSpacing(data.GetSpacing())
    
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
        field.SetTuple1(index,val)
        
        if xval > dims[0] or yval >dims[1] or zval > dims[2]:
            print (xval,yval,zval)
            
    field.SetTuple1(0,0.0)       
    feature_field.GetPointData().AddArray(field)
        
    return feature_field
    
def segment_feature(fname,confidence_th,size_threshold, tstep, outpath):
    
    idx=0
    cnt=0
        
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
    idx=0
    for i in range(num_segments):
        if  bubble_volumes[i] > size_threshold and i not in(top_indices):
            thresholding2 = vtk.vtkThreshold()
            thresholding2.SetInputData(ug)
            thresholding2.ThresholdBetween(i,i)
            thresholding2.Update()
            single_feature = thresholding2.GetOutput()
            idx=idx+1
            
            feature_field = extract_and_dump_feature_vti(single_feature,data)
            
            outfname = outpath + 'segmented_feature_' + str(i) + '_' + str(tstep) + '.vti'
            write_vti(outfname,feature_field)
            
        if  bubble_volumes[i] > size_threshold:
            cnt=cnt+1

#####################################################################
## TODO: Set Params and Datapaths here
initstep = 100
finalstep = 102
confidence_th = 0.9 
size_threshold=100
initPltTstep = 10000
nbins = [128,16,128]
# ### multiprocess all timesteps
data_path = './out/' ## for MAC (assumes that SLIC_gaussian_distance.py is run before this)
plt_data_path = '/Users/sdutta/Data/MFIX_fcc_plt/' ## for MAC
outpath = './out/'
var_name = 'feature_similarity'

inputfname = []
for file in sorted(os.listdir(data_path)):
    if file.endswith(".vti"):
        inputfname.append(os.path.join(data_path,file))       

## sort by timestep numbers
#inputfname.sort(key=lambda f: int( filter(str.isdigit, f) ) ) ## for python 2
inputfname.sort(key=lambda f: int(re.sub('\D', '', f))) ## for python 3

for i in range(len(inputfname)):

	sim_field = read_vti(inputfname[i])

	## get the timestep from filename
	name_components = inputfname[i].split('_')
	name_components = name_components[2].split('.')
	tstep = name_components[0]

	##############################################
	##Do thresholding and connected components
	sim_field.GetPointData().SetActiveScalars(var_name)
	thresholding = vtk.vtkThreshold()
	thresholding.ThresholdByUpper( confidence_th )
	thresholding.SetInputData(sim_field)
	thresholding.Update()

	seg = vtk.vtkConnectivityFilter()
	seg.SetInputConnection(thresholding.GetOutputPort())
	seg.SetExtractionModeToAllRegions()
	seg.ColorRegionsOn()
	seg.Update()
	outfile = outpath +  'slic_segmented_' + str(tstep) + '.vtu'
	write_vtu(outfile,seg.GetOutput())

	## filter all the particles
	plt_fname = plt_data_path + 'fcc' + str(initPltTstep+i*100)
	particle_data = read_plt(plt_fname)

	filtered_particles = filter_particles(particle_data,sim_field,var_name,confidence_th)
	out_fname = outpath + 'filtered_particles_' + str(tstep) + '.vtp'
	write_vtp(out_fname, filtered_particles)

	outp = outpath + '/segmented_volumes/'
	segment_feature(inputfname[i],confidence_th,size_threshold, tstep, outp)


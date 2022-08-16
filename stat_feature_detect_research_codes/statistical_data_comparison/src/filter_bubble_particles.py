## This script runs through all the feature similarity fields, loads corresponding plt particle fields and
## filters the particles which are enclosed inside bubbles. For each timestep a vtp file with the filtererd
## particles are stored.
#####################################################################################
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


#####################################################################
## TODO: Set Params and Datapaths here
confidence_th = 0.92
size_threshold=100
initPltTstep = 10000
nbins = [128,16,128]

data_path = '../out/mfix_local_grid/' ## (assumes that SLIC_gaussian_distance.py is run before this)
plt_data_path = '/Users/sdutta/Data/MFIX_fcc_plt/' ## for MAC
outpath = '../out/isolated_particles/'
var_name = 'feature_similarity'

inputfname = []
for file in sorted(os.listdir(data_path)):
    if file.endswith(".vti"):
        inputfname.append(os.path.join(data_path,file))       

## sort by timestep numbers
#inputfname.sort(key=lambda f: int( filter(str.isdigit, f) ) ) ## for python 2
inputfname.sort(key=lambda f: int(re.sub('\D', '', f))) ## for python 3

print (inputfname)

for i in range(len(inputfname)):

	sim_field = read_vti(inputfname[i])

	## get the timestep from filename
	name_components = inputfname[i].split('_')
	name_components = name_components[2].split('.')
	tstep = name_components[0]

	###############################################
	## filter all the particles
	tstep_number = initPltTstep+i*100

    if (tstep_number < 10000):
        plt_tstep = str(0) + str(tstep_number)
    else:
        plt_tstep = str(tstep_number)
    
    plt_fname = plt_data_path + 'fcc' + plt_tstep
    #print (plt_fname)
    particle_data = read_plt(plt_fname)
	filtered_particles = filter_particles(particle_data,sim_field,var_name,confidence_th)
	out_fname = outpath + 'filtered_particles_' + str(tstep) + '.vtp'
	write_vtp(out_fname, filtered_particles)


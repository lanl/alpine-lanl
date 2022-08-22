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

def construct_vtk_imageData(origin, dimensions, spacing):
    localDataset = vtk.vtkImageData()
    localDataset.SetOrigin(origin)
    localDataset.SetDimensions(dimensions)
    localDataset.SetSpacing(spacing)
    return localDataset

def create_histogram_density_field(inparams):

    #nbins = [128, 16, 128] ##for fcc    
    nbins = [128, 4, 64] ## for thin-bed important

    in_file = inparams[0] + inparams[2]
    #outFile = inparams[1] + inparams[2] + '_' + str(nbins[0]) + '.vti'
    outFile = inparams[1] + inparams[2] + '.vti'

    print (in_file)
    
    reader = vtk.vtkAMReXParticlesReader()
    reader.SetPlotFileName(in_file)
    reader.Update()
    
    NBlocks = reader.GetOutput().GetNumberOfBlocks()
    print("total blocks:",NBlocks)
    
    mpData = reader.GetOutput().GetBlock(0)
    numPieces = mpData.GetNumberOfPieces()
    print("total pieces:", numPieces)
    
    ## compute global bound from the data for each time step
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    zmin = []
    zmax = []
    for i in range(numPieces):
        piece_bound = mpData.GetPiece(i).GetBounds()
        xmin.append(piece_bound[0])
        xmax.append(piece_bound[1])
        ymin.append(piece_bound[2])
        ymax.append(piece_bound[3])
        zmin.append(piece_bound[4])
        zmax.append(piece_bound[5])
        
    bound_min = [np.min(xmin),np.min(ymin),np.min(zmin)] 
    bound_max = [np.max(xmax),np.max(ymax),np.max(zmax)]

    ## If use fix bounds for fcc data for all timesteps
    #     bound_min = [7.44e-5, 7.44e-5, 7.44e-5] 
    #     bound_max = [0.15, 0.0031256, 0.0507256]

    print (bound_min, bound_max)

    pc_cnt = 0
    tot_pts = 0

    for i in range(numPieces):
        if numPieces==1:
            data = mpData
        else:
            data = mpData.GetPiece(i)

        if data is not None:
            pts = data.GetPoints()
            tot_pts = tot_pts + data.GetNumberOfPoints()

    print("Total size:", tot_pts)

    feat_arr = np.zeros((tot_pts, 3), dtype='float')
    cur_count = 0
    for i in range(numPieces):
        if numPieces==1:
            data = mpData
        else:
            data = mpData.GetPiece(i)

        if data is not None:
            pts = data.GetPoints()
            local_pts = data.GetNumberOfPoints()

            for i in range(local_pts):
                loc = pts.GetPoint(i)
                feat_arr[cur_count + i, :] = np.asarray(loc)
            cur_count = cur_count + data.GetNumberOfPoints()

    print("read all the points, Total size:", np.shape(feat_arr))

    # ## Dump a vtp file with all points
    # allpoints = vtk.vtkPoints()
    # for ii in range(len(feat_arr)):
    #     #print (feat_arr[ii])
    #     allpoints.InsertNextPoint(feat_arr[ii])

    # allpdata = vtk.vtkPolyData()
    # allpdata.SetPoints(allpoints)

    # writer1 = vtk.vtkXMLPolyDataWriter()
    # writer1.SetFileName("fcc_allpoints.vtp")
    # writer1.SetInputData(allpdata)
    # writer1.Write()
    
    ## Compute the histogram
    H, edges = np.histogramdd(feat_arr, bins=nbins, range=[[bound_min[0],bound_max[0]],[bound_min[1],bound_max[1]],[bound_min[2],bound_max[2]]])

    ## xdel, ydel, zdel represents bin width of histogram and also works as spacing for imagedata
    xdel = edges[0][1]-edges[0][0]
    ydel = edges[1][1]-edges[1][0]
    zdel = edges[2][1]-edges[2][0]
    
    # write out a vti file with the density information
    origin = [bound_min[0]+xdel/2.0,bound_min[1]+ydel/2.0,bound_min[2]+zdel/2.0]
    dimensions = nbins
    spacing = [xdel,ydel,zdel]
    dataset = construct_vtk_imageData(origin,dimensions,spacing)
    dataset.AllocateScalars(vtk.VTK_DOUBLE, 1)
    dims = dataset.GetDimensions()

    # Fill every entry of the image data with density information
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                dataset.SetScalarComponentFromDouble(x, y, z, 0, H[x,y,z])

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(outFile)
    writer.SetInputData(dataset)
    writer.Write()

#########################################################################
##parallel multithreading execution: create density field for all files
data_path = '/Users/sdutta/Data/MFIX_thinbed/'
outpath = '/Users/sdutta/Desktop/thinbed_density_field/'

###################################################################
# Create a pool of worker processes, each able to use a CPU core
#pool = Pool(processes=1)

files = sorted(os.listdir(data_path))
args = [(data_path,outpath,files[i]) for i in range(len(files))] 
# print (args)

# Execute the multiprocess code
#pool.map(create_histogram_density_field, args)

## call in serial mode to compute times
for j in range(len(args)):
    print ('processing file: ' + str(j))
    create_histogram_density_field(args[j])


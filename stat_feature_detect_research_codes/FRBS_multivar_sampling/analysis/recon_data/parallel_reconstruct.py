from mpi4py import MPI
import numpy as np
from sklearn.neighbors import NearestNeighbors
import vtk
from scipy.interpolate import griddata
import sys
import os

comm = MPI.COMM_WORLD

numProcs = comm.size
procId = comm.rank

comm.Barrier()

###############################################################################################
## Isabel

XDIM = 250 
YDIM = 250 
ZDIM = 50
infile = '/Users/sdutta/Codes/multivar_sampling/output/pmi_sampled_isabel_p.vtp'
cur_samp = 'linear' # nearest
var_name = 'Pressure' 
outfile = '/Users/sdutta/Codes/multivar_sampling/output/recon_data/recon_pmi_sampled_isabel_p.vti'
spacing = np.array([1, 1, 1])

###############################################################################################

###############################################################################################
## Asteroid

# XDIM = 300 
# YDIM = 300 
# ZDIM = 300
# infile = '/Users/sdutta/Codes/fuzzy_rule_based_multivar_sampling/output/Asteroid_tev_v02_3_percent_sampled_hist.vtp'
# cur_samp = 'linear' # nearest
# var_name = 'tev' 
# outfile = '/Users/sdutta/Codes/fuzzy_rule_based_multivar_sampling/analysis/recon_data/asteroid_pmi_recon_' + var_name + '_' + cur_samp + '.vti'
# origin = np.array([0, 0, 0])
# spacing = np.array([1, 1, 1])

###############################################################################################


###############################################################################################
## Nyx

# XDIM = 128 
# YDIM = 128 
# ZDIM = 128
# infile = '/Users/sdutta/Codes/multivar_sampling/output/pmi_sampled_prob_nyx.vtp'
# cur_samp = 'linear' # nearest
# var_name = 'logField' 
# outfile = '/Users/sdutta/Codes/multivar_sampling/output/nyx_i1_recon_temperature_' + var_name + '_' + cur_samp + '.vti'
# origin = np.array([0, 0, 0])
# spacing = np.array([4.0236220472440944, 4.0236220472440944, 4.0236220472440944])

###############################################################################################

poly_reader = vtk.vtkXMLPolyDataReader()
poly_reader.SetFileName(infile)
poly_reader.Update()
data = poly_reader.GetOutput()

print("total points: ",data.GetNumberOfPoints(),data.GetNumberOfElements(0))

pts = data.GetPoints()
print(pts.GetPoint(0))

tot_pts = data.GetNumberOfPoints()
feat_arr = np.zeros((tot_pts,3))

print('total points:',tot_pts)

data_vals = np.zeros(tot_pts)

for i in range(tot_pts):
    loc = pts.GetPoint(i)
    feat_arr[i,:] = np.asarray(loc)
    pt_data = data.GetPointData().GetArray(var_name).GetTuple1(i)
    data_vals[i] = pt_data

range_min = np.min(feat_arr,axis=0)
range_max = np.max(feat_arr,axis=0)

print("range:",range_min,range_max)

print("trying send receive")
if procId == 0:
    data = np.arange(100, dtype=np.float64)
    comm.Send(data, dest=1, tag=13)
elif procId == 1:
    data = np.empty(100, dtype=np.float64)
    comm.Recv(data, source=0, tag=13)
print("Done test")

# currently, only divide the xdim
xdel = (int)(XDIM/numProcs)
xstart = procId*xdel
if procId==numProcs-1:
    xend = XDIM
else:
    xend = (procId+1)*xdel

cur_loc = np.zeros(((xend-xstart)*YDIM*ZDIM,3),dtype='float32')

ind = 0
for k in range(ZDIM):
    for j in range(YDIM):
        for i in range(xstart,xend):
            cur_loc[ind,:] = origin + spacing * np.array([i,j,k])
            ind = ind+1

grid_z0 = griddata(feat_arr, data_vals, cur_loc, method=cur_samp)
grid_vals = np.zeros_like(grid_z0)
grid_vals = np.copy(grid_z0)

## do receives
if procId == 0:
    g_grid_z0_3d = np.zeros((ZDIM,YDIM,XDIM),dtype='float32')
    ## set the local data first
    g_grid_z0_3d[:,:,xstart:xend] = grid_vals.reshape((ZDIM,YDIM,xend-xstart))
    for k in range(1,numProcs):
        xstart = k*xdel
        if k==numProcs-1:
            xend = XDIM
        else:
            xend = (k+1)*xdel
        #local_data = np.zeros((xend-xstart)*YDIM*ZDIM,dtype='float32')
        local_data = np.empty((xend-xstart)*YDIM*ZDIM, dtype=np.float64)
        print("receiving from ",k,xend,xstart, "of shape: ",np.shape(local_data))
        comm.Recv(local_data,source=k, tag=k)
        g_grid_z0_3d[:,:,xstart:xend] = local_data.reshape((ZDIM,YDIM,xend-xstart))

## do sends
else:
    #send data to proc 0
    print("sending data of shape:",np.shape(grid_vals),"proc: ",procId)
    comm.Send(grid_vals, dest=0, tag=procId)
print("Done send receives.")

if procId == 0:
    print("writing file:")

    # write to a vti file
    #filename = 'recons_parallel_stitched'+data_set+'_'+samp_method+'_'+cur_samp+'.vti'
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(XDIM, YDIM, ZDIM)

    imageData.SetOrigin(origin)
    imageData.SetSpacing(spacing)


    if vtk.VTK_MAJOR_VERSION <= 5:
        imageData.SetNumberOfScalarComponents(1)
        imageData.SetScalarTypeToDouble()
    else:
        imageData.AllocateScalars(vtk.VTK_DOUBLE, 1)

    dims = imageData.GetDimensions()
    print(dims)
    # Fill every entry of the image data with "2.0"
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                imageData.SetScalarComponentFromDouble(x, y, z, 0, g_grid_z0_3d[z,y,x])

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(outfile)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInputConnection(imageData.GetProducerPort())
    else:
        writer.SetInputData(imageData)
    writer.Write()

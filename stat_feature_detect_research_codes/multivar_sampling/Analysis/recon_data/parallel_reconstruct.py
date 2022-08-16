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

# ###############################################################################################
# ## Isabel
# data_set = 'isabel'
# origin = np.array([0, 0, 0])
# spacing = np.array([1, 1, 1])
# XDIM = 300 
# YDIM = 300 
# ZDIM = 300

# samp_type = 'pmi' # pmi, random
# percentage = 5 # 1,3,5,7,9
# cur_samp = 'linear' # nearest

# var_name1 = 'Pressure'
# var_name2 = 'Velocity'
# infile = '../../output/isabel_sampled/joint_'+samp_type+'_sampled_'+data_set+'_'+var_name1+'_'+var_name2+'_'+str(percentage)+'.vtp'

# outfile1 = '../../output/recon_linear/joint_' + samp_type + '_recon_'+data_set+'_'+var_name1+'_'+cur_samp+'_'+str(percentage)+'.vti'
# outfile2 = '../../output/recon_linear/joint_' + samp_type + '_recon_'+data_set+'_'+var_name2+'_'+cur_samp+'_'+str(percentage)+'.vti'

# # ###############################################################################################

# ###############################################################################################
# ## combustion
# data_set = 'combustion'
# origin = np.array([0, 0, 0])
# spacing = np.array([1, 1, 1])
# XDIM = 240 
# YDIM = 360 
# ZDIM = 60

# samp_type = 'random' # pmi, random
# percentage = 3 # 1,3,5,7,9
# cur_samp = 'linear' # nearest

# var_name1 = 'mixfrac'
# var_name2 = 'Y_OH'
# infile = '../../output/joint_'+samp_type+'_sampled_'+data_set+'_'+var_name1+'_'+var_name2+'_'+str(percentage)+'.vtp'

# outfile1 = '../../output/recon_linear/joint_' + samp_type + '_recon_'+data_set+'_'+var_name1+'_'+cur_samp+'_'+str(percentage)+'.vti'
# outfile2 = '../../output/recon_linear/joint_' + samp_type + '_recon_'+data_set+'_'+var_name2+'_'+cur_samp+'_'+str(percentage)+'.vti'

# ###############################################################################################

###############################################################################################
## Asteroid
data_set = 'asteroid'
origin = np.array([-2300000.0, -500000.0, -1200000.0])
spacing = np.array([15384.615385, 9364.548495, 8026.7558528])
XDIM = 300 
YDIM = 300 
ZDIM = 300

samp_type = 'pmi' # pmi, random
percentage = 3 # 1,3,5,7,9
cur_samp = 'linear' # nearest

var_name1 = 'tev'
var_name2 = 'v02'
infile = '../../output/asteroid_sampled/joint_'+samp_type+'_sampled_'+data_set+'_'+var_name1+'_'+var_name2+'_'+str(percentage)+'.vtp'

outfile1 = '../../output/recon_linear/joint_' + samp_type + '_recon_'+data_set+'_'+var_name1+'_'+cur_samp+'_'+str(percentage)+'.vti'
outfile2 = '../../output/recon_linear/joint_' + samp_type + '_recon_'+data_set+'_'+var_name2+'_'+cur_samp+'_'+str(percentage)+'.vti'

###############################################################################################

def recon_data(var_name,outfile,infile,origin,spacing,XDIM,YDIM,ZDIM):

    poly_reader = vtk.vtkXMLPolyDataReader()
    poly_reader.SetFileName(infile)
    poly_reader.Update()
    data = poly_reader.GetOutput()
    print("total points: ",data.GetNumberOfPoints(),data.GetNumberOfElements(0))

    pts = data.GetPoints()
    print(pts.GetPoint(0))

    tot_pts = data.GetNumberOfPoints()
    feat_arr = np.zeros((tot_pts,3))
    
    data_vals = np.zeros(tot_pts)

    for i in range(tot_pts):
        loc = pts.GetPoint(i)
        feat_arr[i,:] = np.asarray(loc)
        pt_data = data.GetPointData().GetArray(var_name).GetTuple1(i)
        data_vals[i] = pt_data

    range_min = np.min(feat_arr,axis=0)
    range_max = np.max(feat_arr,axis=0)

    print("range:",range_min,range_max)

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
            #print("receiving from ",k,xend,xstart, "of shape: ",np.shape(local_data))
            comm.Recv(local_data,source=k, tag=k)
            g_grid_z0_3d[:,:,xstart:xend] = local_data.reshape((ZDIM,YDIM,xend-xstart))

    ## do sends
    else:
        #send data to proc 0
        #print("sending data of shape:",np.shape(grid_vals),"proc: ",procId)
        comm.Send(grid_vals, dest=0, tag=procId)
    print("Done send receives.")

    if procId == 0:
        print("writing file:")

        # write to a vti file
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(XDIM, YDIM, ZDIM)
        imageData.SetOrigin(origin)
        imageData.SetSpacing(spacing)

        arr = vtk.vtkDoubleArray()
        arr.SetNumberOfTuples(XDIM*YDIM*ZDIM)
        arr.SetName(var_name)

        # Fill every entry of the image data with "2.0"
        index=0
        for z in range(ZDIM):
            for y in range(YDIM):
                for x in range(XDIM):
                    arr.SetTuple1(index,g_grid_z0_3d[z,y,x])
                    index=index+1

        imageData.GetPointData().AddArray(arr)            

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(outfile)
        writer.SetInputData(imageData)
        writer.Write()

####################################################################################################

if procId == 0:
    print 'reconstructing: ' + var_name1
recon_data(var_name1,outfile1,infile,origin,spacing,XDIM,YDIM,ZDIM)

if procId == 0:
    print 'reconstructing: ' + var_name2
recon_data(var_name2,outfile2,infile,origin,spacing,XDIM,YDIM,ZDIM)

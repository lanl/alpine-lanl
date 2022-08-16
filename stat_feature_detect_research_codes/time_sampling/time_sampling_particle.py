import vtk
import numpy as np
import sys
import os
import math
import time
from numpy import linalg as LA
from matplotlib import pyplot as plt
###############################################

def compute_entropy_global(datain):
    
    nbins = [32, 4, 4]
    
    NBlocks = datain.GetNumberOfBlocks()
    # print("total blocks:",NBlocks)
    
    mpData = datain.GetBlock(0)
    numPieces = mpData.GetNumberOfPieces()
    # print("total pieces:", numPieces)

    ## Hard coded fixed bounds for now for hourglass data bound from inputs file
    bound_min = [0,0,0] ## smooth-hg
    bound_max = [0.9,0.4,0.4] ## smooth-hg
    #print (bound_min, bound_max)

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

    #print("Read all the points, Total size:", np.shape(feat_arr))
    
    ## Compute the histogram
    H, edges = np.histogramdd(feat_arr, bins=nbins, range=[[bound_min[0],bound_max[0]],[bound_min[1],bound_max[1]],[bound_min[2],bound_max[2]]])

    ## Now compute entropy
    entropy = 0.0
    for i in range(np.shape(H)[0]):
        for j in range(np.shape(H)[1]):
            for k in range(np.shape(H)[2]):
                prob = H[i,j,k]/tot_pts
                if prob > 0:
                    entropy = entropy -  prob*math.log(prob, 2)

    return entropy

def compute_entropy_local(datain):
    
    nbins = [32, 4, 4]
    
    NBlocks = datain.GetNumberOfBlocks()
    # print("total blocks:",NBlocks)
    
    mpData = datain.GetBlock(0)
    numPieces = mpData.GetNumberOfPieces()
    # print("total pieces:", numPieces)
    
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
    #print (bound_min, bound_max)

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

    #print("Read all the points, Total size:", np.shape(feat_arr))
    
    ## Compute the histogram
    H, edges = np.histogramdd(feat_arr, bins=nbins, range=[[bound_min[0],bound_max[0]],[bound_min[1],bound_max[1]],[bound_min[2],bound_max[2]]])

    ## Now compute entropy
    entropy = 0.0
    for i in range(np.shape(H)[0]):
        for j in range(np.shape(H)[1]):
            for k in range(np.shape(H)[2]):
                prob = H[i,j,k]/tot_pts
                if prob > 0:
                    entropy = entropy -  prob*math.log(prob, 2)

    return entropy

def compute_entropy_global_local_velmag(datain):
    
    nbins = [32, 4, 4, 16] ## for smooth hg

    NBlocks = datain.GetNumberOfBlocks()
    mpData = datain.GetBlock(0)
    numPieces = mpData.GetNumberOfPieces()

    tot_pts = 0
    for i in range(numPieces):
        if numPieces==1:
            data = mpData
        else:
            data = mpData.GetPiece(i)

        if data is not None:
            pts = data.GetPoints()
            tot_pts = tot_pts + data.GetNumberOfPoints()

    feat_arr = np.zeros((tot_pts, 4), dtype='float')
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
                velx = data.GetPointData().GetArray('velx').GetTuple1(i)
                vely = data.GetPointData().GetArray('vely').GetTuple1(i)
                velz = data.GetPointData().GetArray('velz').GetTuple1(i)
                vel_norm = LA.norm([velx,vely,velz])

                feat_arr[cur_count + i, :] = np.asarray([loc[0],loc[1],loc[2],vel_norm])

            cur_count = cur_count + data.GetNumberOfPoints()

    #print("Read all the points, Total size:", np.shape(feat_arr))

    vel_max = np.max(feat_arr[:,3])
    vel_min = np.min(feat_arr[:,3])
    #print (vel_min,vel_max)

    ## Hard coded fixed bounds for global computation
    bound_min_particle = [0,0,0] ## smooth-hg
    bound_max_particle = [0.9,0.4,0.4] ## smooth-hg
    # # ## Compute the histogram
    H, edges = np.histogramdd(feat_arr, bins=nbins, range=[[bound_min_particle[0],bound_max_particle[0]],[bound_min_particle[1],bound_max_particle[1]],[bound_min_particle[2],bound_max_particle[2]], [vel_min,vel_max] ])
    ## Now compute entropy
    entropy1 = 0.0
    prob=0
    for i in range(np.shape(H)[0]):
        for j in range(np.shape(H)[1]):
            for k in range(np.shape(H)[2]):
                for l in range(np.shape(H)[3]):
                    prob = H[i,j,k,l]/tot_pts
                    if prob > 0:
                        entropy1 = entropy1 -  prob*math.log(prob, 2)


    #############################################################
    ## compute local particle bound from the data for each time step
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
    bound_min_local = [np.min(xmin),np.min(ymin),np.min(zmin)] 
    bound_max_local = [np.max(xmax),np.max(ymax),np.max(zmax)]

    # # ## Compute the histogram
    H1, edges1 = np.histogramdd(feat_arr, bins=nbins, range=[[bound_min_local[0],bound_max_local[0]],[bound_min_local[1],bound_max_local[1]],[bound_min_local[2],bound_max_local[2]], [vel_min,vel_max] ])
    ## Now compute entropy
    entropy2 = 0.0
    prob=0
    for i in range(np.shape(H1)[0]):
        for j in range(np.shape(H1)[1]):
            for k in range(np.shape(H1)[2]):
                for l in range(np.shape(H1)[3]):
                    prob = H1[i,j,k,l]/tot_pts
                    if prob > 0:
                        entropy2 = entropy2 -  prob*math.log(prob, 2)

    return entropy1,entropy2


#################################################################
data_path = '/Users/sdutta/Data/hourglass_data/plt_data/plt_data/'

initT = 0
time_window = 100

entropy_list_global = []
entropy_list_local = []
entropy_list_global_velmag = []
entropy_list_local_velmag = []

file1 = open("entropy_global_exashallow.txt","a") 
file2 = open("entropy_local_exashallow.txt","a") 
file3 = open("entropy_global_velmag_exashallow.txt","a") 
file4 = open("entropy_local_velmag_exashallow.txt","a") 

tstep = initT
for (rootDir, subDirs, files) in os.walk(data_path):
    for subDir in sorted(subDirs):
        if subDir.startswith('plt'):
            print ('processing: ' + subDir)
            in_file = data_path + subDir

            reader = vtk.vtkAMReXParticlesReader()
            reader.SetPlotFileName(in_file)
            reader.Update()

            ####
            entropy = compute_entropy_global(reader.GetOutput())
            entropy_list_global.append(entropy)
            print ('gloabl entropy is: '+str(entropy))
            file1.writelines([ str(tstep), "," ,str(entropy), "\n" ] ) 

            #####
            entropy = compute_entropy_local(reader.GetOutput())
            entropy_list_local.append(entropy)
            print ('local entropy is: '+str(entropy))
            file2.writelines([ str(tstep), "," ,str(entropy), "\n" ] )

            ####
            entropy1,entropy2 = compute_entropy_global_local_velmag(reader.GetOutput())
            entropy_list_global_velmag.append(entropy1)
            entropy_list_local_velmag.append(entropy2)
            print ('global + velmag entropy is: '+str(entropy1))
            print ('local + velmag entropy is: '+str(entropy2))
            file3.writelines([ str(tstep), "," ,str(entropy1), "\n" ] )
            file4.writelines([ str(tstep), "," ,str(entropy2), "\n" ] )

            tstep = tstep + time_window
############################################


file1.close()     
file2.close()
file3.close()
file4.close()


# np.savetxt('entropy_global_exashallow.txt',entropy_list_global)
# np.savetxt('entropy_local_exashallow.txt',entropy_list_local)
# np.savetxt('entropy_global_velmag_exashallow.txt',entropy_list_global_velmag)
# np.savetxt('entropy_local_velmag_exashallow.txt',entropy_list_local_velmag)


import math
import os
import sys
import numpy as np
from tqdm import trange
from scipy.spatial import distance

import vtk
from vtk.util import numpy_support as VN
import random
from numpy import linalg as LA

from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import seaborn as sns
from termcolor import colored

class DataManager(object):
    
    def __init__(self, fname, target_var_id):

        self.fname = fname
        self.varId = target_var_id
        self.f_extension = os.path.splitext(os.path.basename(self.fname))[1]
        self.supported_extenstions = ['.vtp', '.vtk', '.vtu', '.vti']
        self.varName = ""
        self.data = []
        self.grad = []
        self.dataSpacing = []
        self.dataOrigin = []
        self.dataExtent = []

        if self.f_extension in self.supported_extenstions and self.varId > -1:
            self.update()
        else:
            print("check supported file extensions and/or variable id")
            raise SystemExit


    # def __del__(self):
    #     return 1; 
    #     #print("Cluster ", self.no ," Destructor called") 

    def update(self):        

        if self.f_extension=='.vtp':
            reader = vtk.vtkXMLPolyDataReader()
        elif self.f_extension=='.vtk':
            reader = vtk.vtkGenericDataObjectReader()
        elif self.f_extension=='.vtu':
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif self.f_extension=='.vti':
            reader = vtk.vtkXMLImageDataReader()
        
        reader.SetFileName(self.fname)
        reader.Update()
        vtk_data = reader.GetOutput()

        self.XDIM, self.YDIM, self.ZDIM = vtk_data.GetDimensions()
        self.dataSpacing = np.asarray(vtk_data.GetSpacing())
        self.dataOrigin = np.asarray(vtk_data.GetOrigin())
        self.dataExtent = np.asarray(vtk_data.GetExtent())
        self.varName = vtk_data.GetPointData().GetArrayName(self.varId)

        self.data = vtk_data.GetPointData().GetArray(self.varId)
        self.data = VN.vtk_to_numpy(self.data)
        self.data = self.data.reshape((self.ZDIM, self.YDIM, self.XDIM))

        #self.data = np.copy(vals)

        self.min = np.min(self.data.flatten())
        self.max = np.max(self.data.flatten())

        self.grad_min = []
        self.grad_max = []


    def get_datafield(self):
        return self.data;

    def get_dimension(self):
        return self.XDIM, self.YDIM, self.ZDIM;

    def get_spacing(self):
        return self.dataSpacing;

    def get_origin(self):
        return self.dataOrigin;

    def get_extent(self):
        return self.dataExtent;

    def get_varName(self):
        return self.varName;

    def get_gradfield(self):
        return self.grad;

    def __str__(self):
        return "\n \
            FILENAME:{} \n \
            XDIM:{} \t YDIM:{} \t ZDIM:{}\n \
            VAR NAME:{} \n \
            SPACING:{} \n \
            ORIGIN:{} \n \
            EXTENT:{} \n \
            SHAPE:{} " \
            .format(self.fname, self.XDIM, self.YDIM, self.ZDIM, self.varName, \
                self.dataSpacing, self.dataOrigin, self.dataExtent, self.data.shape)

    def __repr__(self):
        return self.__str__()

    def getMin(self): 
        return self.min

    def getMax(self): 
        return self.max

    def getGradMin(self): 
        return self.grad_min

    def getGradMax(self): 
        return self.grad_max

    def get_acceptance_histogram(self, sampling_ratio, nbins=32):

        global_hist_count, global_bin_edges = np.histogram(self.data.flatten(), bins=nbins)
        #print('bin edges:',global_bin_edges)

        samples = self.XDIM * self.YDIM * self.ZDIM

        frac = sampling_ratio
        tot_samples = frac*samples
        print('looking for', tot_samples,'samples')
        
        # create a dictionary first
        my_dict = dict() 
        ind = 0
        for i in global_hist_count:
            my_dict[ind] = i
            ind = ind + 1
        #print(my_dict)

        sorted_count = sorted(my_dict, key=lambda k: my_dict[k])
        #print("here:",sorted_count)

        ## now distribute across bins
        target_bin_vals = int(tot_samples/nbins)
        #print('ideal cut:',target_bin_vals)
        new_count = np.copy(global_hist_count)
        ind = 0
        remain_tot_samples = tot_samples
        for i in sorted_count:
            if my_dict[i]>target_bin_vals:
                val = target_bin_vals
            else:
                val = my_dict[i]
                remain = target_bin_vals-my_dict[i]
            new_count[i]=val
            #print(new_count[i], target_bin_vals)
            ind = ind + 1
            remain_tot_samples = remain_tot_samples-val
            if ind < nbins:
                target_bin_vals = int(remain_tot_samples/(nbins-ind))
        #print(new_count)  
        #print(count)  
        #acceptance_hist = new_count/count
        
        acceptance_hist = np.zeros_like(new_count,dtype='float32')
        for i in range(nbins):
            if global_hist_count[i]==0:
                acceptance_hist[i]=0
            else:
                acceptance_hist[i]=new_count[i]/global_hist_count[i]
        #print("acceptance histogram created:", acceptance_hist)
        return acceptance_hist


    def get_acceptance_histogram_grad(self, sampling_ratio, nbins=32):

        # compute gradient on the 3D data

        vals_3d_grad_mag = LA.norm(np.gradient(self.data),axis=0)

        self.grad = vals_3d_grad_mag

        self.grad_min = np.min(vals_3d_grad_mag)
        self.grad_max = np.max(vals_3d_grad_mag)

        #np.shape(vals_3d_grad_mag)

        grad_hist, edges = np.histogram(np.ndarray.flatten(vals_3d_grad_mag),bins=nbins)

        samples = np.size(vals_3d_grad_mag)

        acceptance_hist_grad = np.zeros_like(grad_hist)
        acceptance_hist_grad_prob = np.zeros_like(grad_hist,dtype='float32')

        dist_counts = int(samples*sampling_ratio)
        remain_counts = dist_counts
        cur_count = 0
        for ii in range(nbins-1,-1,-1):## looping from the most grad to least
            #print(ii)
            if remain_counts<grad_hist[ii]:
                cur_count=remain_counts
            else:
                cur_count=grad_hist[ii]
            acceptance_hist_grad[ii]=cur_count
            remain_counts = remain_counts-cur_count
            if grad_hist[ii]>0.00000005:
                acceptance_hist_grad_prob[ii] = acceptance_hist_grad[ii]/grad_hist[ii]

        return acceptance_hist_grad_prob


class BlockManager(object):

    def __init__(self, partitionType, partitionParameteList):
        
        self.partitionType = partitionType 

        if partitionType=='regular':
            #extract parameters
            self.XBLOCK = np.int(partitionParameteList[0])
            self.YBLOCK = np.int(partitionParameteList[1])
            self.ZBLOCK = np.int(partitionParameteList[2])

            self.XDIM = np.int(partitionParameteList[3])
            self.YDIM = np.int(partitionParameteList[4])
            self.ZDIM = np.int(partitionParameteList[5])

            self.d_origin = partitionParameteList[6]
            self.d_spacing = partitionParameteList[7]
            self.d_extent = partitionParameteList[8]

            self.XB_COUNT = np.int(self.XDIM/self.XBLOCK)
            self.YB_COUNT = np.int(self.YDIM/self.YBLOCK)
            self.ZB_COUNT = np.int(self.ZDIM/self.ZBLOCK)
            
            

            #self.regularPartition(data)

        elif partitionType=='kdtree':
            #extract parameters
            #kdtreePartition(data)
            print("kdtree not implemented")
            raise SystemExit
        elif partitionType=='slic':
            #extract parameters
            #slicPartition(data)
            print("slic not implemented")
            raise SystemExit
        else:
            print("unsupported partitioning")
            raise SystemExit

    def partition(self, data):

        if self.partitionType=='regular':
            print("partitioning removed")
            raise SystemExit

        elif self.partitionType=='kdtree':
            #extract parameters
            #kdtreePartition(data)
            print("kdtree not implemented")
            raise SystemExit
        elif self.partitionType=='slic':
            #extract parameters
            #slicPartition(data)
            print("slic not implemented")
            raise SystemExit
        else:
            print("unsupported partitioning")
            raise SystemExit

    

    def numberOfBlocks(self):
        return self.XB_COUNT * self.YB_COUNT * self.ZB_COUNT;

    def get_blockData(self, data, bid):
        # block_size = self.XBLOCK * self.YBLOCK * self.ZBLOCK
        # block_data = np.zeros((block_size,))

        fid_s, tx_s, ty_s, tz_s = self.func_block_2_full(bid, 0)

        block_data = data[tz_s:tz_s+self.ZBLOCK, ty_s:ty_s+self.YBLOCK, tx_s:tx_s+self.XBLOCK]
        
        # for lid in range(block_size):
        #     fid, tx, ty, tz = self.func_block_2_full(bid, lid)
        #     block_data[lid] = data[tz][ty][tx]

        return block_data.flatten() 


    def func_xyz_2_fid(self, x, y, z):
        if x >= self.XDIM or y >= self.YDIM or z >= self.ZDIM or x < 0 or y < 0 or z < 0:
            print("out of bound")
            raise SystemExit
        return z*self.YDIM*self.XDIM + y*self.XDIM + x

    def func_fid_2_xyz(self, fid):
        if fid >= self.XDIM*self.YDIM*self.ZDIM or fid < 0:
            print("out of bound")
            raise SystemExit
        fid = np.int(fid)
        z = np.int(fid / (self.XDIM * self.YDIM))
        fid -= np.int(z * self.XDIM * self.YDIM)
        y = np.int(fid / self.XDIM)
        x = np.int(fid % self.XDIM)
        return x, y, z

    def func_block_2_full(self, bid, local_id=0):
        
        
        if bid >= self.XB_COUNT*self.YB_COUNT*self.ZB_COUNT or bid < 0:
            print("out of bound")
            raise SystemExit
        bid = np.int(bid)
        bz = np.int(bid / (self.XB_COUNT * self.YB_COUNT))
        bid -= np.int(bz * self.XB_COUNT * self.YB_COUNT)
        by = np.int(bid / self.XB_COUNT)
        bx = np.int(bid % self.XB_COUNT)
        
        x = np.int(bx*self.XBLOCK)
        y = np.int(by*self.XBLOCK)
        z = np.int(bz*self.XBLOCK)
        
        
        if local_id >= self.XBLOCK*self.YBLOCK*self.ZBLOCK or local_id < 0:
            print("[local id] out of bound")
            raise SystemExit
            
        local_id = np.int(local_id)
        local_z = np.int(local_id / (self.XBLOCK * self.YBLOCK))
        local_id -= np.int(local_z * self.XBLOCK * self.YBLOCK)
        local_y = np.int(local_id / self.XBLOCK)
        local_x = np.int(local_id % self.XBLOCK)
        
        fx = x + local_x
        fy = y + local_y
        fz = z + local_z
                        
        
        fid = fz*self.YDIM*self.XDIM + fy*self.XDIM + fx
        
        return fid, fx, fy, fz

    def func_logical_2_physical_location(self, x, y, z):
        px = x*self.d_spacing[0]
        py = y*self.d_spacing[1]
        pz = z*self.d_spacing[2]
        
        return px, py, pz

    def func_physical_2_logical_location(self, x, y, z):
        lx = x/self.d_spacing[0]
        ly = y/self.d_spacing[1]
        lz = z/self.d_spacing[2]
        
        return lx, ly, lz


class SampleManager(object ):

    def __init__(self):
        print("Initialized")
        # self.samplingMethod = samplingMethod
        # self.xdim = xdim
        # self.ydim = ydim
        # self.zdim = zdim

        #self.stencil

    def set_global_properties(self, global_acceptance_hist, global_min, global_max):
        self.global_acceptance_hist = global_acceptance_hist
        self.global_min = global_min
        self.global_max = global_max

    def set_global_properties_grad(self, global_acceptance_hist_grad, grad_global_min, grad_global_max):
        self.global_acceptance_hist_grad = global_acceptance_hist_grad
        self.global_grad_min = grad_global_min
        self.global_grad_max = grad_global_max

    def global_grad_based_sampling(self, grad, dim):
        
        #data = np.asarray(data)
        grad_flat = np.asarray(np.ndarray.flatten(grad))
        nbins = np.size(self.global_acceptance_hist_grad)
        samples = np.size(grad_flat)
        stencil = np.zeros_like(grad_flat)
        prob_vals = np.zeros_like(grad_flat)
        rand_vals = np.zeros_like(grad_flat)

        # decide which bin this point goes 
        #print('Sampling:reloading this module ')
        #t_bin_delta = (self.global_max - self.global_min)/nbins
        t_bin_edges = np.linspace(self.global_grad_min, self.global_grad_max,nbins+1)
        #print('bin edges:',t_bin_edges)
        t_xids = np.digitize(grad_flat,t_bin_edges)
        t_xids-=1
        t_xids[t_xids>=nbins]=nbins-1
        prob_vals = self.global_acceptance_hist_grad[t_xids]
        rand_vals = np.random.uniform(0, 1, samples)

        stencil[rand_vals<prob_vals]=1

        # #generate the void distribution using the global_bin_edges information
        # void_values = vals_np[stencil < 0.5]
        # global_void_hist = np.histogram(void_values, bins=global_bin_edges)

        # #generate the local histogram of the block
        # local_void_hist = np.histogram(void_values, bins=local_bins)

        return stencil

    def global_hist_grad_rand_based_sampling(self, data, dim):
        
        data = np.asarray(data)
        nbins = np.size(self.global_acceptance_hist)
        samples = np.size(data)
        stencil = np.zeros_like(data)
        prob_vals = np.zeros_like(data)
        rand_vals = np.zeros_like(data)
        grad_power = 1

        ## add local gradient
        
        # compute gradient on the 3D data
        #dim = [32,32,32]
        vals_3d = data.reshape((dim[2],dim[1],dim[0]))
        #print(np.shape(vals_3d))

        vals_3d_grad_mag = LA.norm(np.gradient(vals_3d),axis=0)
        vals_3d_grad_mag_flattened = np.ndarray.flatten(vals_3d_grad_mag)

        #print("Starting 2D acceptance histogram creation")

        hist_2d, xedges, yedges = np.histogram2d(data,vals_3d_grad_mag_flattened,bins=nbins,range=[[self.global_min, self.global_max], [np.min(vals_3d_grad_mag), np.max(vals_3d_grad_mag)]])

        count = np.sum(hist_2d,axis=1)
        new_count = self.global_acceptance_hist*count

        ybin_centers = np.zeros(nbins)

        for bc in range(nbins):
            ybin_centers[bc] = (yedges[bc]+yedges[bc+1])/2.0

        bc_probs = ybin_centers/np.sum(ybin_centers)

        cur_bc = np.zeros_like(bc_probs)
        
        ## create 2D acceptance histogram
        ## for each bin in 1D histogram, we now have the counts
        ## now start assiging points to 2D bins for each bin of 1D bin, start with largest gradients

        acceptance_hist_2D = np.zeros_like(hist_2d)
        acceptance_hist_2D_prob = np.zeros_like(hist_2d,dtype='float32')
        #('printing',np.sum(count[0]), np.sum(hist_2d[0,:]),new_count[0])

        # print('printing 1',count[0],hist_2d[:,0], hist_2d[0,:],new_count[0])

        for jj in range(nbins):
            if new_count[jj]>0.0000001:
                dist_counts = new_count[jj]
                # print("acceptance hist: ",acceptance_hist[jj])
                inds = np.where(hist_2d[jj,:]>0.000001)
                cur_bc = np.zeros_like(bc_probs)
                cur_bc[inds] = ybin_centers[inds]
                if np.sum(cur_bc)>0.00000001:
                    cur_bc_prob = np.power(cur_bc,1.0/grad_power)/np.sum(cur_bc)
                    # print("cur_bc_prob: ", cur_bc_prob, "indices: ",inds)
                    remain_counts = dist_counts
                    cur_count = 0
                    samp_vals = cur_bc_prob*dist_counts
                    samp_vals = np.minimum(samp_vals,hist_2d[jj,:])
                    # print("wanted samples: ",new_count[jj], "getting sample: ", samp_vals, np.sum(samp_vals), "from: ",hist_2d[jj,:],np.sum(hist_2d[jj,:]))
                    remain_samp_count = new_count[jj]-np.sum(samp_vals)
                    remaining_samp_vals = hist_2d[jj,:]-samp_vals
                    # print("remaining_samp_vals:",remaining_samp_vals)
                    ## sort again
                    # create a dictionary first
                    my_dict1 = dict() 
                    ind = 0
                    for i in remaining_samp_vals:
                        my_dict1[ind] = i
                        ind = ind + 1
                    # print(my_dict1)
                    new_sorted_count = sorted(my_dict1, key=lambda k: my_dict1[k])
                    # print("sorted count:",new_sorted_count)
                    ind =0
                    target_bin_vals = remain_samp_count/(nbins-ind)
                    for i in new_sorted_count:
                        if my_dict1[i]>target_bin_vals:
                            val = target_bin_vals
                        else:
                            val = my_dict1[i]
                            #break
                            # remain = target_bin_vals-my_dict[i]
                        samp_vals[i]=samp_vals[i] + val
                        # print(samp_vals[i], target_bin_vals)
                        ind = ind + 1
                        remain_samp_count = remain_samp_count-val
                        if ind < nbins:
                            target_bin_vals = remain_samp_count/(nbins-ind)
                    # print("Getting samples now: ",samp_vals,np.sum(samp_vals))
                    ## distribute these many samples over the bins equally
                    for ii in range(nbins):
                        if hist_2d[jj,ii]>0.00000001:
                            acceptance_hist_2D_prob[jj,ii] = samp_vals[ii]/hist_2d[jj,ii]
        
        acceptance_hist_2D_prob[acceptance_hist_2D_prob>1.0]=1.0

        # decide which bin this point goes 
        #print('Sampling:reloading this module ')
        #t_bin_delta = (self.global_max - self.global_min)/nbins
        t_bin_edges = np.linspace(self.global_min, self.global_max,nbins+1)
        #print('bin edges:',t_bin_edges)
        t_xids = np.digitize(data,t_bin_edges)
        t_xids-=1
        t_xids[t_xids>=nbins]=nbins-1

        t_yids = np.digitize(vals_3d_grad_mag_flattened,yedges)
        t_yids-=1
        t_yids[t_yids>=nbins]=nbins-1

        #prob_vals = self.global_acceptance_hist[t_xids]
        prob_vals = acceptance_hist_2D_prob[t_xids,t_yids]
        rand_vals = np.random.uniform(0, 1, samples)

        stencil[rand_vals<prob_vals]=1

        # #generate the void distribution using the global_bin_edges information
        # void_values = vals_np[stencil < 0.5]
        # global_void_hist = np.histogram(void_values, bins=global_bin_edges)

        # #generate the local histogram of the block
        # local_void_hist = np.histogram(void_values, bins=local_bins)

        return stencil

    def global_hist_grad_based_sampling(self, data, dim):
        
        data = np.asarray(data)
        nbins = np.size(self.global_acceptance_hist)
        samples = np.size(data)
        stencil = np.zeros_like(data)
        prob_vals = np.zeros_like(data)
        rand_vals = np.zeros_like(data)

        ## add local gradient
        #dim = [32,32,32]
        vals_3d = data.reshape((dim[2],dim[1],dim[0]))
        #print(np.shape(vals_3d))

        #vals_3d_grad = np.gradient(vals_3d)
        #print(np.shape(vals_3d_grad))

        #vals_3d_grad_mag = LA.norm(vals_3d_grad,axis=0)

        vals_3d_grad_mag = LA.norm(np.gradient(vals_3d),axis=0)
        vals_3d_grad_mag_flattened = np.ndarray.flatten(vals_3d_grad_mag)

        hist_2d, xedges, yedges = np.histogram2d(data,vals_3d_grad_mag_flattened,bins=nbins,range=[[self.global_min, self.global_max], [np.min(vals_3d_grad_mag), np.max(vals_3d_grad_mag)]])

        count = np.sum(hist_2d,axis=1)
        new_count = self.global_acceptance_hist*count

        ## create 2D acceptance histogram
        ## for each bin in 1D histogram, we now have the counts
        ## now start assiging points to 2D bins for each bin of 1D bin, start with largest gradients

        acceptance_hist_2D = np.zeros_like(hist_2d)
        acceptance_hist_2D_prob = np.zeros_like(hist_2d,dtype='float32')
        #print('printing',np.sum(count[0]), np.sum(hist_2d[0,:]),new_count[0])

        # print('printing 1',count[0],hist_2d[:,0], hist_2d[0,:],new_count[0])

        for jj in range(nbins):
            #print('printing',np.sum(count[jj]), np.sum(hist_2d[jj,:]),new_count[jj])
            if new_count[jj]>0.0000001:
                dist_counts = new_count[jj]
                remain_counts = dist_counts
                cur_count = 0
                for ii in range(nbins-1,-1,-1):## looping from the most grad to least
                    if hist_2d[jj,ii]>0.0000001:
                        if remain_counts<hist_2d[jj,ii]:
                            cur_count=remain_counts
                            acceptance_hist_2D[jj,ii]=cur_count
                            remain_counts = remain_counts-cur_count
                            acceptance_hist_2D_prob[jj,ii] = acceptance_hist_2D[jj,ii]/hist_2d[jj,ii]
                            break
                        else:
                            cur_count=hist_2d[jj,ii]
                            acceptance_hist_2D[jj,ii]=cur_count
                            remain_counts = remain_counts-cur_count
                            acceptance_hist_2D_prob[jj,ii] = acceptance_hist_2D[jj,ii]/hist_2d[jj,ii]
        #acceptance_hist_2D_prob = acceptance_hist_2D/hist_2d
                #print('orig',dist_counts,'remain:',remain_counts)
        # decide which bin this point goes 
        #print('Sampling:reloading this module ')
        #t_bin_delta = (self.global_max - self.global_min)/nbins
        t_bin_edges = np.linspace(self.global_min, self.global_max,nbins+1)
        #print('bin edges:',t_bin_edges)
        t_xids = np.digitize(data,t_bin_edges)
        t_xids-=1
        t_xids[t_xids>=nbins]=nbins-1

        t_yids = np.digitize(vals_3d_grad_mag_flattened,yedges)
        t_yids-=1
        t_yids[t_yids>=nbins]=nbins-1

        #prob_vals = self.global_acceptance_hist[t_xids]
        prob_vals = acceptance_hist_2D_prob[t_xids,t_yids]
        rand_vals = np.random.uniform(0, 1, samples)

        stencil[rand_vals<prob_vals]=1

        # #generate the void distribution using the global_bin_edges information
        # void_values = vals_np[stencil < 0.5]
        # global_void_hist = np.histogram(void_values, bins=global_bin_edges)

        # #generate the local histogram of the block
        # local_void_hist = np.histogram(void_values, bins=local_bins)

        return stencil

    def global_hist_based_sampling(self, data):
        
        data = np.asarray(data)
        nbins = np.size(self.global_acceptance_hist)
        samples = np.size(data)
        stencil = np.zeros_like(data)
        prob_vals = np.zeros_like(data)
        rand_vals = np.zeros_like(data)

        # decide which bin this point goes 
        #print('Sampling:reloading this module ')
        #t_bin_delta = (self.global_max - self.global_min)/nbins
        t_bin_edges = np.linspace(self.global_min, self.global_max,nbins+1)
        #print('bin edges:',t_bin_edges)
        t_xids = np.digitize(data,t_bin_edges)
        t_xids-=1
        t_xids[t_xids>=nbins]=nbins-1
        prob_vals = self.global_acceptance_hist[t_xids]
        rand_vals = np.random.uniform(0, 1, samples)

        stencil[rand_vals<prob_vals]=1

        # #generate the void distribution using the global_bin_edges information
        # void_values = vals_np[stencil < 0.5]
        # global_void_hist = np.histogram(void_values, bins=global_bin_edges)

        # #generate the local histogram of the block
        # local_void_hist = np.histogram(void_values, bins=local_bins)

        return stencil

    def rand_sampling(self, data, sampling_ratio):
        data = np.asarray(data)
        samples = np.size(data)
        stencil = np.zeros_like(data)
        
        tot_samples = sampling_ratio*samples
        sel_locations = np.random.choice(np.int(samples), np.int(tot_samples), replace=False)
        
        temp_ones = np.ones_like(sel_locations)

        np.put(stencil, sel_locations, temp_ones)

        # for loc in sel_locations:
        #     stencil[np.int(loc)] = 1.0

        #print('selected ', np.sum(stencil),' random samples')
        
        return stencil   

    ##TODO: refactor
    def hist_based_sampling(self, sampling_ratio, vals_np, nbins=32):
        ## vals_np holds the RTData values
        ## now apply sampling algorithms
        samples = np.size(vals_np)
        stencil = np.zeros_like(vals_np)
        prob_vals = np.zeros_like(vals_np)
        rand_vals = np.zeros_like(vals_np)
        ## Histogram based feature driven sampling
        count, bin_edges = np.histogram(vals_np,bins=nbins)

        #print(count)

        frac = sampling_ratio
        tot_samples = frac*samples
        print('looking for',tot_samples,'samples')
        # create a dictionary first
        my_dict = dict() 
        ind = 0
        for i in count:
            my_dict[ind] = i
            ind = ind + 1
        #print(my_dict)

        sorted_count = sorted(my_dict, key=lambda k: my_dict[k])
        #print("here:",sorted_count)

        ## now distribute across bins
        target_bin_vals = int(tot_samples/nbins)
        #print('ideal cut:',target_bin_vals)
        new_count = np.copy(count)
        ind = 0
        remain_tot_samples = tot_samples
        for i in sorted_count:
            if my_dict[i]>target_bin_vals:
                val = target_bin_vals
            else:
                val = my_dict[i]
                remain = target_bin_vals-my_dict[i]
            new_count[i]=val
            #print(new_count[i], target_bin_vals)
            ind = ind + 1
            remain_tot_samples = remain_tot_samples-val
            if ind < nbins:
                target_bin_vals = int(remain_tot_samples/(nbins-ind))
        #print(new_count)  
        #print(count)  
        #acceptance_hist = new_count/count
        
        acceptance_hist = np.zeros_like(new_count,dtype='float32')
        for i in range(nbins):
            if count[i]==0:
                acceptance_hist[i]=0
            else:
                acceptance_hist[i]=new_count[i]/count[i]
        print("acceptance histogram:",acceptance_hist)


        # In[ ]:

        bound_min = np.min(vals_np)
        bound_max = np.max(vals_np)

        # decide which bin this point goes to
        for i in range(samples):
            loc = vals_np[i]
            x_id = int(nbins * (loc - bound_min) / (bound_max - bound_min))
            if x_id == nbins:
                x_id = x_id - 1
            prob_vals[i]=acceptance_hist[x_id]
            # generate a random number
            rand_vals[i] = random.uniform(0, 1)
        #     if rand_vals[i]<prob_vals[i]:
        #         stencil[i] = 1

        stencil[rand_vals<prob_vals]=1

        return stencil

    def get_void_histogram(self, data, stencil, nbins=10):
        
        samples = np.size(data)

        if samples != np.size(stencil):
            print("stencil size doesn't match data to sample")
            raise SystemExit

        void_values = data[stencil < 0.5]
        
        void_hist = np.histogram(void_values, bins=nbins)
        
        delta = void_hist[1][1] - void_hist[1][0]
        bin_left_edge = void_hist[1][0]

        return void_hist[0], bin_left_edge, delta


    def get_samples(self, block_data, stencil):

        samples = np.size(block_data)

        if samples != np.size(stencil):
            print("stencil size doesn't match data to sample")
            raise SystemExit
        
        sampled_data = block_data[stencil > 0.5]

        sampled_local_id = np.where(stencil > 0.5)[0]

        #sampled_output = np.vstack((sampled_fid, sampled_data))

        return sampled_local_id, sampled_data


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        #p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, ia[i])

class ReconstructionManager(object):

    def __init__(self):
        print("Initialized")

    def func_lid_2_xyz(self, fid, XDIM, YDIM, ZDIM):
        if fid >= XDIM*YDIM*ZDIM or fid < 0:
            print("out of bound")
            raise SystemExit
        fid = np.int(fid)
        z = np.int(fid / (XDIM * YDIM))
        fid -= np.int(z * XDIM * YDIM)
        y = np.int(fid / XDIM)
        x = np.int(fid % XDIM)
        return x, y, z

    def sample_from_histogram(self, hist, nos):
        hist_count = hist[0]
        bins = hist[1]
        
        bin_midpoints = bins[:-1] + np.diff(bins)/2
        cdf = np.cumsum(hist_count)
        cdf = cdf / cdf[-1]
        values = np.random.rand(nos)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = bin_midpoints[value_bins]
        return random_from_cdf

    def reconstruction_1(self, block_size, sampled_data, sampled_lid, vhist_count, vhist_ble, vhist_delta,block_dims):

        recon_block = np.full((block_size,), np.nan)

        np.put(recon_block, sampled_lid, sampled_data)

        nan_count = np.sum(np.isnan(recon_block))
        nan_location = np.where(np.isnan(recon_block))[0]

        #create void histogram 
        vhist = [0, 1]
        vhist[0] = vhist_count
        v_nbins = np.size(vhist[0])
        vhist[1] = np.arange(v_nbins+1,)
        vhist[1] = vhist[1]*vhist_delta + vhist_ble 

        vhist_sval =  self.sample_from_histogram(vhist, nan_count)
        np.put(recon_block, nan_location, vhist_sval)

        if np.sum(np.isnan(recon_block)) != 0:
            print("reconstruction error: still have nan values")
            raise SystemExit

        # for i in range(block_size):
        #     if np.isnan(recon_block[i]):
        #         recon_block[i] = np.mean(self.sample_from_histogram(void_histogram, 1))

        return recon_block

    def reconstruction_2(self, block_size, sampled_data, sampled_lid, vhist_count, vhist_ble, vhist_delta,block_dims):

        recon_block = np.full((block_size,), np.nan)

        np.put(recon_block, sampled_lid, sampled_data)

        nan_count = np.sum(np.isnan(recon_block))
        nan_location = np.where(np.isnan(recon_block))[0]

        #create void histogram 
        vhist = [0, 1]
        vhist[0] = vhist_count
        v_nbins = np.size(vhist[0])
        vhist[1] = np.arange(v_nbins+1,)
        vhist[1] = vhist[1]*vhist_delta + vhist_ble 

        # bins = vhist[1]
        # bin_midpoints = bins[:-1] + np.diff(bins)/2
        # vhist_sval = random.choices(bin_midpoints, vhist[0], k = nan_count)
        vhist_sval =  self.sample_from_histogram(vhist, nan_count)
        np.put(recon_block, nan_location, vhist_sval)

        if np.sum(np.isnan(recon_block)) != 0:
            print("reconstruction error: still have nan values")
            raise SystemExit

        # for i in range(block_size):
        #     if np.isnan(recon_block[i]):
        #         recon_block[i] = np.mean(self.sample_from_histogram(void_histogram, 1))

        return recon_block

    def reconstruction_3(self, block_size, sampled_data, sampled_lid, vhist_count, vhist_ble, vhist_delta,block_dims):

        recon_block = np.full((block_size,), np.nan)

        np.put(recon_block, sampled_lid, sampled_data)

        nan_count = np.sum(np.isnan(recon_block))
        nan_location = np.where(np.isnan(recon_block))[0]
        if not np.size(nan_location):
            return recon_block

        #create void histogram 
        vhist = [0, 1]
        vhist[0] = vhist_count
        v_nbins = np.size(vhist[0])
        vhist[1] = np.arange(v_nbins+1,)
        vhist[1] = vhist[1]*vhist_delta + vhist_ble 

        bins = vhist[1]
        
        bin_midpoints = bins[:-1] + np.diff(bins)/2

        #block_dims = [32,32,32]

        ## compute nearest neighbors of nanlocations from sampled locations
        n_neighbors = 1
        # first unravel
        #print('total samples',np.shape(sampled_lid))
        unraveled_sample_lid = np.transpose(np.unravel_index(sampled_lid,block_dims))
        unraveled_nan_locations = np.transpose(np.unravel_index(nan_location,block_dims))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(unraveled_sample_lid)
        distances, indices = nbrs.kneighbors(unraveled_nan_locations)

        ## do argsort of the distances
        sorted_sequence = np.argsort(distances[:,0])

        ## find the nearest neighbor values for each nan location
        NNs = sampled_data[np.asarray(indices)[:,0]]

        #print(distances,np.min(distances),np.max(distances),np.shape(distances))
        #print(sorted_sequence)

        #print(vhist_count, np.sum(vhist_count))

        # digitize the NNs to later lookup
        t_xids = np.digitize(NNs,vhist[1])
        t_xids-=1
        t_xids[t_xids>=v_nbins]=v_nbins-1

        ## in a for loop
        for ind in sorted_sequence:
            if vhist_count[t_xids[ind]]>=1:
                # assign this value
                recon_block[nan_location[ind]] = NNs[ind]
                # reduce the count
                vhist_count[t_xids[ind]]-=1
            else:
                # find non-zero locations of vhist[0], count of local histogram and 
                # which bin centers of vhist[1] correspond to this
                non_zero_indices = np.nonzero(vhist_count)
                non_zero_indices = non_zero_indices[0][:]
                bin_filts = bin_midpoints[non_zero_indices]

                # find nearest match of digitized NNs from this bin centers; can use 'if' for the same id
                val, val_indx = find_nearest(bin_filts,NNs[ind])

                # set recon block with this value
                recon_block[nan_location[ind]] = val

                # decrease count of the local histogram by 1
                #print(np.shape(non_zero_indices),val_indx,end=' ')
                vhist_count[non_zero_indices[val_indx]] -=1

        if np.sum(vhist_count) != 0:
            print("histogram error: check the code! Sum is",np.sum(vhist_count))
            raise SystemExit

        if np.sum(np.isnan(recon_block)) != 0:
            print("reconstruction error: still have nan values")
            raise SystemExit

        # for i in range(block_size):
        #     if np.isnan(recon_block[i]):
        #         recon_block[i] = np.mean(self.sample_from_histogram(void_histogram, 1))

        return recon_block

    def reconstruction_4(self, block_size, sampled_data, sampled_lid, vhist_count, vhist_ble, vhist_delta,block_dims):

        recon_block = np.full((block_size,), np.nan)

        np.put(recon_block, sampled_lid, sampled_data)

        nan_count = np.sum(np.isnan(recon_block))
        nan_location = np.where(np.isnan(recon_block))[0]

        if not np.size(nan_location):
            return recon_block

        #create void histogram 
        vhist = [0, 1]
        vhist[0] = vhist_count
        v_nbins = np.size(vhist[0])
        vhist[1] = np.arange(v_nbins+1,)
        vhist[1] = vhist[1]*vhist_delta + vhist_ble 

        bins = vhist[1]
        
        bin_midpoints = bins[:-1] + np.diff(bins)/2

        #block_dims = [32,32,32]

        ## compute nearest neighbors of nanlocations from sampled locations
        n_neighbors = 1
        # first unravel
        unraveled_sample_lid = np.transpose(np.unravel_index(sampled_lid,block_dims))
        unraveled_nan_locations = np.transpose(np.unravel_index(nan_location,block_dims))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(unraveled_sample_lid)
        distances, indices = nbrs.kneighbors(unraveled_nan_locations)

        ## do argsort of the distances
        sorted_sequence = np.argsort(distances[:,0])

        ## find the nearest neighbor values for each nan location
        NNs = sampled_data[np.asarray(indices)[:,0]]

        #print(distances,np.min(distances),np.max(distances),np.shape(distances))
        #print(sorted_sequence)

        #print(vhist_count, np.sum(vhist_count))

        # digitize the NNs to later lookup
        t_xids = np.digitize(NNs,vhist[1])
        t_xids-=1
        t_xids[t_xids>=v_nbins]=v_nbins-1

        runs, vals = rle(t_xids)

        ## in a for loop
        for ind in sorted_sequence:
            if vhist_count[t_xids[ind]]>=1:
                # assign this value
                recon_block[nan_location[ind]] = NNs[ind]
                # reduce the count
                vhist_count[t_xids[ind]]-=1
            else:
                # find non-zero locations of vhist[0], count of local histogram and 
                # which bin centers of vhist[1] correspond to this
                non_zero_indices = np.nonzero(vhist_count)
                non_zero_indices = non_zero_indices[0][:]
                bin_filts = bin_midpoints[non_zero_indices]

                # find nearest match of digitized NNs from this bin centers; can use 'if' for the same id
                val, val_indx = find_nearest(bin_filts,NNs[ind])

                # set recon block with this value
                recon_block[nan_location[ind]] = val

                # decrease count of the local histogram by 1
                #print(np.shape(non_zero_indices),val_indx,end=' ')
                vhist_count[non_zero_indices[val_indx]] -=1

        if np.sum(vhist_count) != 0:
            print("histogram error: check the code! Sum is",np.sum(vhist_count))
            raise SystemExit

        if np.sum(np.isnan(recon_block)) != 0:
            print("reconstruction error: still have nan values")
            raise SystemExit

        # for i in range(block_size):
        #     if np.isnan(recon_block[i]):
        #         recon_block[i] = np.mean(self.sample_from_histogram(void_histogram, 1))

        return recon_block

    # griddata based interpolation
    def reconstruction_5(self, block_size, sampled_data, sampled_lid, vhist_count, vhist_ble, vhist_delta,block_dims): 

        recon_block = np.full((block_size,), np.nan)

        np.put(recon_block, sampled_lid, sampled_data)

        nan_count = np.sum(np.isnan(recon_block))
        nan_location = np.where(np.isnan(recon_block))[0]

        if not np.size(nan_location):
            return recon_block

        cur_samp = 'nearest'

        #print(np.shape(sampled_lid), np.shape(sampled_data), np.shape(recon_block), np.shape(nan_location))

        tot_pts = np.size(sampled_lid)
        feat_arr = np.zeros((tot_pts,3))

        #print('total points:',tot_pts)

        data_vals = np.zeros(tot_pts)

        for i in range(tot_pts):
            feat_arr[i,:] = self.func_lid_2_xyz(sampled_lid[i], block_dims[0], block_dims[1],block_dims[2])

        cur_loc = np.zeros((block_size,3),dtype='double')

        ind = 0
        for k in range(block_dims[2]):
            for j in range(block_dims[1]):
                for i in range(block_dims[0]):
                    cur_loc[ind,:] = np.array([i,j,k])
                    ind = ind+1
        
        grid_z0 = griddata(feat_arr, sampled_data, cur_loc, method=cur_samp)

        #grid_z0 = griddata(sampled_lid, sampled_data, recon_block, method=cur_samp)
        #grid_z0_3d = grid_z0.reshape((block_dims[2],block_dims[1],block_dims[0]))

        return grid_z0

    # griddata based interpolation
    def reconstruction_6(self, block_size, sampled_data, sampled_lid, vhist_count, vhist_ble, vhist_delta,block_dims): 

        recon_block = np.full((block_size,), np.nan)

        np.put(recon_block, sampled_lid, sampled_data)

        nan_count = np.sum(np.isnan(recon_block))
        nan_location = np.where(np.isnan(recon_block))[0]

        if not np.size(nan_location):
            return recon_block

        cur_samp = 'linear'

        #print(np.shape(sampled_lid), np.shape(sampled_data), np.shape(recon_block), np.shape(nan_location))

        tot_pts = np.size(sampled_lid)
        feat_arr = np.zeros((tot_pts,3))

        #print('total points:',tot_pts)

        data_vals = np.zeros(tot_pts)

        for i in range(tot_pts):
            feat_arr[i,:] = self.func_lid_2_xyz(sampled_lid[i], block_dims[0], block_dims[1],block_dims[2])

        cur_loc = np.zeros((block_size,3),dtype='double')

        ind = 0
        for k in range(block_dims[2]):
            for j in range(block_dims[1]):
                for i in range(block_dims[0]):
                    cur_loc[ind,:] = np.array([i,j,k])
                    ind = ind+1
        
        grid_z0 = griddata(feat_arr, sampled_data, cur_loc, method=cur_samp)

        #grid_z0 = griddata(sampled_lid, sampled_data, recon_block, method=cur_samp)
        #grid_z0_3d = grid_z0.reshape((block_dims[2],block_dims[1],block_dims[0]))

        return grid_z0

    # def reconstruction_2(self, block_xyz, sampled_data, void_histogram):

    #     recon_block = np.full((len(block_xyz),), np.nan)

    #     for sd in sampled_data:
    #         recon_block[sd[0]] = sd[2]

    #     s_xyz_array = np.zeros((len(sampled_data),3))
    #     for i in range(len(sampled_data)):
    #         s_xyz_array[i] = sampled_data[i][1]

    #     nn = NearestNeighbors()
    #     nn.fit(s_xyz_array)

    #     for i in range(len(block_xyz)):
    #         if np.isnan(recon_block[i]):
    #             nn_dis, nn_id =nn.kneighbors([block_xyz[i]], 5)

    #             recon_block[i] = sampled_data[nn_id[0][0]][2]

    #     return recon_block.tolist()










##Some Utility/Viz functions
def showImage(image,  name, cm_name):
    #%matplotlib inline
    
    #print(colored(name,'red'))

    if cm_name == 'default':
        cm_name = 'viridis'

    im = plt.imshow(image, interpolation='nearest', cmap=plt.cm.get_cmap(cm_name))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.colorbar(im, fraction=0.020, pad=0.04)
    plt.title(name)
    plt.show()

def showImage_clim(image,  name, cm_name, minVal, maxVal):
    #%matplotlib inline
    
    #print(colored(name,'red'))

    if cm_name == 'default':
        cm_name = 'viridis'

    im = plt.imshow(image, interpolation='nearest', cmap=plt.cm.get_cmap(cm_name))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.colorbar(im, fraction=0.020, pad=0.04)
    plt.title(name)
    plt.clim(minVal,maxVal)
    plt.show()
    
def showImage_cat(image,name, numCat, minVal, maxVal):
    #%matplotlib inline  

    im = plt.imshow(image, interpolation='nearest', cmap=plt.cm.get_cmap('tab20', numCat))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.colorbar(im, fraction=0.020, pad=0.04,  ticks=range(numCat))
    plt.title(name)
    plt.clim(minVal - 0.5, maxVal + 0.5)
    plt.show()
    
def showImage_cat_cm_name(image,name, numCat, minVal, maxVal, cm_name):
    #%matplotlib inline  

    im = plt.imshow(image, interpolation='nearest', cmap=plt.cm.get_cmap(cm_name, numCat))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.colorbar(im, fraction=0.020, pad=0.04,  ticks=range(numCat))
    plt.title(name)
    plt.clim(minVal - 0.5, maxVal + 0.5)
    plt.show()
    
def showBWImage(image):
    #%matplotlib inline  

    plt.imshow(image, interpolation='nearest',cmap='gray')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.colorbar()
    plt.show()
    
def showImage_autoSize(image):
    #%matplotlib inline 

    plt.imshow(image,  aspect='auto', interpolation='none')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.colorbar()
    plt.show()

def showImage_autoSize_clim(image, minValue, maxValue):
    #%matplotlib inline 

    plt.imshow(image,  aspect='auto', interpolation='none')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.colorbar()
    plt.clim(minValue,maxValue)
    plt.show()

def getMinMax(dat):
    my_max = sys.float_info.min
    my_min = sys.float_info.max
    
    for y in range(0,ydim):
        for x in range(0,xdim):
            if not np.isnan(dat[y][x]):
                val = dat[y][x]
                if val >= my_max:
                    my_max = val
                if val <= my_min:
                    my_min = val
    return my_min,my_max 

def get_acceptance_hist_1d(sampling_ratio, samples, global_hist_count ):
    
    nbins = np.size(global_hist_count)
    frac = sampling_ratio
    tot_samples = frac*samples
    print('looking for',tot_samples,'samples')
    
    # create a dictionary first
    my_dict = dict() 
    ind = 0
    for i in global_hist_count:
        my_dict[ind] = i
        ind = ind + 1
    #print(my_dict)

    sorted_count = sorted(my_dict, key=lambda k: my_dict[k])
    #print("here:",sorted_count)

    ## now distribute across bins
    target_bin_vals = int(tot_samples/nbins)
    #print('ideal cut:',target_bin_vals)
    new_count = np.copy(global_hist_count)
    ind = 0
    remain_tot_samples = tot_samples
    for i in sorted_count:
        if my_dict[i]>target_bin_vals:
            val = target_bin_vals
        else:
            val = my_dict[i]
            remain = target_bin_vals-my_dict[i]
        new_count[i]=val
        #print(new_count[i], target_bin_vals)
        ind = ind + 1
        remain_tot_samples = remain_tot_samples-val
        if ind < nbins:
            target_bin_vals = int(remain_tot_samples/(nbins-ind))
    #print(new_count)  
    #print(count)  
    #acceptance_hist = new_count/count
    
    acceptance_hist = np.zeros_like(new_count,dtype='float32')
    for i in range(nbins):
        if global_hist_count[i]==0:
            acceptance_hist[i]=0
        else:
            acceptance_hist[i]=new_count[i]/global_hist_count[i]
    print("acceptance histogram created:",acceptance_hist)
    return acceptance_hist


##Use for testing the functions
if __name__ == '__main__':

    dm = DataManager("/Users/shazarika/projects/data/alpine_milestone-data/nyx_converted_ascii.vtk", 2)
    print(dm)
    data = dm.get_datafield();
    print(data.shape)
    print(dm.get_valueAt(50,50,50))
    print(data[50][50][50])

    bm = BlockManager('regular', [50,50,50])
    print(bm.numberOfBlocks())


    
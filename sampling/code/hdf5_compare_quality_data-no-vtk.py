#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata
import sys
import os


# In[2]:


blockname = '510'
infilename = 'density_full.cycle_000001/domain_000'+blockname+'.hdf5'
outfilename = 'out'+blockname+'.vtp'


# In[3]:


f = h5py.File(infilename, 'r')

list(f.keys())

np_data = np.asarray(f['fields']['Density']['values'])
print(np_data)

f.close()


# In[4]:


f = h5py.File(infilename, 'r')

list(f['coordsets']['coords'].keys())
print(np.asarray(f['coordsets']['coords']['spacing']))
print(np.asarray(f['coordsets']['coords']['origin']))

spacing_x = np.asarray(f['coordsets']['coords']['spacing']['dx'])
spacing_y = np.asarray(f['coordsets']['coords']['spacing']['dy'])
spacing_z = np.asarray(f['coordsets']['coords']['spacing']['dz'])
spacing = np.ndarray.flatten(np.asarray([spacing_x, spacing_y, spacing_z]))
origin_x = np.asarray(f['coordsets']['coords']['origin']['x'])
origin_y = np.asarray(f['coordsets']['coords']['origin']['y'])
origin_z = np.asarray(f['coordsets']['coords']['origin']['z'])
origin = np.ndarray.flatten(np.asarray([origin_x, origin_y, origin_z]))
print(origin,spacing)


# In[5]:


np.shape(np_data)
np_data_3d = np.reshape(np_data, (130,130,130))
# remove ghost layer
np_data_3d_ng = np_data_3d[1:129,1:129,1:129]


# In[6]:


print(np.min(np_data_3d),np.min(np_data_3d_ng),np.max(np_data_3d),np.max(np_data_3d_ng))
print(np.shape(np_data_3d_ng))


# In[7]:


## read the sampled data

insampfilename = 'density_sampled.cycle_000001/domain_000'+blockname+'.hdf5'


# In[9]:


f = h5py.File(insampfilename, 'r')

np_data = np.asarray(f['fields']['Density']['values'])
print(np_data)

np_coordsX = np.asarray(f['coordsets']['coords']['values']['x'])
print(np_coordsX)
np_coordsY = np.asarray(f['coordsets']['coords']['values']['y'])
print(np_coordsY)
np_coordsZ = np.asarray(f['coordsets']['coords']['values']['z'])
print(np_coordsZ)

f.close()

tot_points = np.size(np_coordsX)


# In[12]:


XDIM = 128 # 250 # 512
YDIM = 128 # 250 # 512
ZDIM = 128  # 50 # 512

data_set = 'nyx' # 'isabel_pressure_10_percent' # 'nyx_5_percent_'
samp_method = 'hist' # random, hist , hist_grad, kdtree_histgrad_random
cur_samp = 'linear' #'nearest'#'linear'

feat_arr = np.zeros((tot_points,3))

print('total points:',tot_points)

data_vals = np.zeros(tot_points)

    
data_vals = np_data
    
feat_arr[:,0] = np_coordsX
feat_arr[:,1] = np_coordsY
feat_arr[:,2] = np_coordsZ

range_min = np.min(feat_arr,axis=0)
range_max = np.max(feat_arr,axis=0)

print("range:",range_min,range_max)

cur_loc = np.zeros((XDIM*YDIM*ZDIM,3),dtype='double')

ind = 0
for k in range(ZDIM):
    for j in range(YDIM):
        for i in range(XDIM):
            cur_loc[ind,:] = origin + spacing * np.array([i,j,k])
            ind = ind+1

#grid_z0 = griddata(feat_arr, data_vals, cur_loc, method=cur_samp)
grid_z0 = griddata(feat_arr, data_vals, cur_loc, method='nearest')
grid_z1 = griddata(feat_arr, data_vals, cur_loc, method=cur_samp)
# check nan elements
print('total nan elements:',np.count_nonzero(np.isnan(grid_z1)), 'out of:', tot_points)
grid_z1[np.isnan(grid_z1)]=grid_z0[np.isnan(grid_z1)]
grid_z0 = grid_z1


# In[17]:


# print some quality statistics
orig_data = np_data_3d_ng.flatten()
recons_data = grid_z0

rmse = np.sqrt(np.mean((recons_data-orig_data)**2))
print('RMSE:',rmse)

prmse = np.sqrt(np.mean(((recons_data-orig_data)/orig_data)**2))
print('PRMSE:',prmse)


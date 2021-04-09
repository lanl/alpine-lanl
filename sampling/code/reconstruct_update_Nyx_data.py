import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata
import sys
import os


# domain information for 64^3 Nyx data
origin_array_list = [
    np.array([0.0,0.0,0.0]),
    np.array([14.245014245, 0.0, 0.0]),
    np.array([0.0, 14.245014245, 0.0]),
    np.array([14.245014245, 14.245014245, 0.0]),
    np.array([0.0,0.0,14.245014245]),
    np.array([14.245014245, 0.0, 14.245014245]),
    np.array([0.0, 14.245014245, 14.245014245]),
    np.array([14.245014245, 14.245014245, 14.245014245])
]

spacing = np.array([0.44515669515625,0.44515669515625,0.44515669515625])


offset_array_list = [
    [0,0,0],
    [32, 0, 0],
    [0, 32, 0],
    [32, 32, 0],
    [0,0,32],
    [32, 0, 32],
    [0, 32, 32],
    [32, 32, 32]
]


def getreconData(blocknum, samp_id = "1d"):
    infilename = '2dsampling/data/full/density_full.cycle_000003/domain_00000'+ str(blocknum)+'.hdf5'

    f = h5py.File(infilename, 'r')
    # print(list(f.keys()))

    # print(list(f['coordsets']['coords'].keys()))
    # print(np.asarray(f['coordsets']['coords']))

    np_data = np.asarray(f['fields']['Density']['values'])

    np_data_3d = np.reshape(np_data, (34,34,34))
    np_data_3d_ng = np_data_3d[1:33,1:33,1:33]


    # print(np.min(np_data_3d),np.min(np_data_3d_ng),np.max(np_data_3d),np.max(np_data_3d_ng))
    # print(np.shape(np_data_3d_ng))

    insampfilename = '2dsampling/data/'+samp_id+'/density_sampled.cycle_000003/domain_00000'+ str(blocknum)+'.hdf5'

    samp_f = h5py.File(insampfilename, 'r')

    np_data_s = np.asarray(samp_f['fields']['Density']['values'])
    #print(np_data_s.shape)

    np_coordsX = np.asarray(samp_f['coordsets']['coords']['values']['x'])
    #print(np_coordsX)
    np_coordsY = np.asarray(samp_f['coordsets']['coords']['values']['y'])
    #print(np_coordsY)
    np_coordsZ = np.asarray(samp_f['coordsets']['coords']['values']['z'])
    #print(np_coordsZ)


    tot_points = np.size(np_coordsX)

    XDIM = 32 # 250 # 512
    YDIM = 32 # 250 # 512
    ZDIM = 32  # 50 # 512

    data_set = 'nyx' # 'isabel_pressure_10_percent' # 'nyx_5_percent_'
    samp_method = 'hist' # random, hist , hist_grad, kdtree_histgrad_random
    cur_samp = 'linear' #'nearest'#'linear'

    feat_arr = np.zeros((tot_points,3))

    print('total points:',tot_points)

    data_vals = np.zeros(tot_points)

    data_vals = np_data_s

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
                cur_loc[ind,:] = origin_array_list[blocknum] + spacing * np.array([i,j,k])
                ind = ind+1


    grid_z0 = griddata(feat_arr, data_vals, cur_loc, method='nearest')
    grid_z1 = griddata(feat_arr, data_vals, cur_loc, method=cur_samp)


    # check nan elements
    print('total nan elements:',np.count_nonzero(np.isnan(grid_z1)), 'out of:', XDIM*YDIM*ZDIM)

    grid_z1[np.isnan(grid_z1)]=grid_z0[np.isnan(grid_z1)]
    grid_z0 = grid_z1

    # print some quality statistics
    orig_data = np_data_3d_ng.flatten()
    recons_data = grid_z0

    rmse = np.sqrt(np.mean((recons_data-orig_data)**2))
    print('RMSE:',rmse)

    prmse = np.sqrt(np.mean(((recons_data-orig_data)/orig_data)**2))
    print('PRMSE:',prmse)

    f.close()
    samp_f.close()
    
    return np.reshape(recons_data, (32,32,32)), np_data_3d_ng


new_array_recon = np.zeros((64,64,64))

for i in range(8):
    rdata, odata = getreconData(i, samp_id="2d")
    print(offset_array_list[i])
    new_array_recon[offset_array_list[i][0]:offset_array_list[i][0]+32, offset_array_list[i][1]:offset_array_list[i][1]+32,offset_array_list[i][2]:offset_array_list[i][2]+32] = rdata
    
    

infilename = 'plt00000_2D_sample_reconstruction.h5'
h5f = h5py.File(infilename, 'r+')
list(h5f['native_fields'].keys())
hf_data = h5f['native_fields']['dark_matter_density']
hf_data[...] = new_array_recon
print(np.min(hf_data),np.max(hf_data) )
h5f.close()


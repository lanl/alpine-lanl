#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import seaborn as sns
from termcolor import colored
import vtk
from sklearn.neighbors import NearestNeighbors
from vtk.util import numpy_support as VN
import os
import time
import sys
sys.path.append('.')
import FeatureSampler as FS
import importlib
importlib.reload(FS)
import pickle
import argparse
#import lzma
import shutil
import pymp
from pathlib import Path

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def dump_with_pickle(pyObj, filename):
    fobj1 = open(filename, 'wb')
    pickle.dump(pyObj, fobj1)
    fobj1.close()
    
def load_with_pickle(filename):
    fobj1 = open(filename, 'rb')
    pyObj = pickle.load(fobj1)
    fobj1.close()
    return pyObj

def run(infile, fb_sr,rand_sr, sampling_method, nthreads, out_folder, ghist_params_list):

    #sampling_method_val = np.int(enum_dict[sampling_method])
    #print('sampling_method_val=',sampling_method_val)

    print('Reading data.')
    dm = FS.DataManager(infile, 0) 

    data = dm.get_datafield()

    XDIM, YDIM, ZDIM = dm.get_dimension()
    d_spacing = dm.get_spacing()
    d_origin = dm.get_origin()
    d_extent = dm.get_extent()
    print(dm)

    vhist_nbins = ghist_params_list[0]
    ghist_nbins = ghist_params_list[1]

    #rand_sr = 0.0
    #fb_sr = 0.005
    sampling_rate = fb_sr+rand_sr

    #get the global acceptance histogram (provide sample rate) and min/max
    ghist = dm.get_acceptance_histogram(fb_sr, ghist_nbins)
    gmin = dm.getMin()
    gmax = dm.getMax()

    blk_dims = ghist_params_list[2:]

    bm_paramter_list = [blk_dims[0], blk_dims[1], blk_dims[2], XDIM, YDIM, ZDIM, d_origin, d_spacing, d_extent]

    bm = FS.BlockManager('regular', bm_paramter_list)

    nob = bm.numberOfBlocks()

    sm = FS.SampleManager()


    sm.set_global_properties(ghist, gmin, gmax)

    if sampling_method=='grad':
        t0 = time.time()
        ghist_grad = dm.get_acceptance_histogram_grad(fb_sr, ghist_nbins)
        gmin_grad = dm.getGradMin()
        gmax_grad = dm.getGradMax()

        sm.set_global_properties_grad(ghist_grad, gmin_grad, gmax_grad)

        grad = dm.get_gradfield()
        print('Time taken for grad computation %.2f secs' %(time.time()-t0))


    list_sampled_lid = []
    list_sampled_data = []

    array_void_hist = np.zeros((nob, vhist_nbins))
    array_ble = np.zeros((nob,))
    array_delta = np.zeros((nob,))

    ## for pymp utilization

    t0 = time.time()
    offset = 0.0000005
    bd_dims = [int(XDIM/blk_dims[0]+offset),int(YDIM/blk_dims[1]+offset),int(ZDIM/blk_dims[2]+offset)]
    #blk_dims = [32,32,32]
    #sampling_method = 'hist'
    freq=1
    whichBlock = 0
    numPieces = len(range(whichBlock,nob,freq))
    multiblock = vtk.vtkMultiBlockDataSet()
    multiblock.SetNumberOfBlocks(numPieces)
    pc_cnt = 0
    tot_points = 0

    print('Starting block processing. Total blocks=',nob, bd_dims)

    if not os.path.exists("vtu_outputs/sampled_"+sampling_method+'_pymp/'):
        os.makedirs("vtu_outputs/sampled_"+sampling_method+'_pymp/')

    tot_points = pymp.shared.array(nob, dtype='int32')
    array_void_hist = pymp.shared.array((nob, vhist_nbins), dtype=np.int64)
    array_ble = pymp.shared.array(nob, dtype='float64')
    array_delta = pymp.shared.array(nob, dtype='float64')

    list_sampled_lid = pymp.shared.list(nob*[None])
    list_sampled_data = pymp.shared.list(nob*[None])

    with pymp.Parallel(nthreads) as p:
        for bid in p.range(nob):
            block_data = bm.get_blockData(data, bid)
            #print(bid,'.',end='')
            
            if sampling_method=='hist':
                fb_stencil = sm.global_hist_based_sampling(block_data)
            elif sampling_method=='hist_grad': 
                fb_stencil = sm.global_hist_grad_based_sampling(block_data,blk_dims)
            elif sampling_method=='hist_grad_rand':
                fb_stencil = sm.global_hist_grad_rand_based_sampling(block_data,blk_dims)
            elif sampling_method=='grad':
                block_grad_data = bm.get_blockData(grad, bid)
                fb_stencil = sm.global_grad_based_sampling(block_grad_data,blk_dims)
            else:
                print('Unknown sampling method; Not implemented yet.')
            rand_stencil = sm.rand_sampling(block_data, rand_sr)
            
            comb_stencil = fb_stencil + rand_stencil
            comb_stencil = np.where(comb_stencil > 1, 1, comb_stencil)

            # pick at least one sample
            if np.sum(comb_stencil)==0:
                comb_stencil[0] = 1 

            ## create boundary points
            dim = blk_dims
            A = [0,dim[0]-1]
            B = [0,dim[1]-1]
            C = [0,dim[2]-1]
            # j_vals = ((xval,yval,zval) for xval in A for yval in B for zval in C)
            j_vals = (((zval * dim[0] * dim[1]) + (yval * dim[0]) + xval) for xval in A for yval in B for zval in C)
            j_list = np.asarray(list(j_vals))
            comb_stencil[j_list]=1
            
            void_hist, ble, delta = sm.get_void_histogram(block_data, comb_stencil, vhist_nbins)
                
            #sampled_fid, sampled_data = sm.get_samples(block_data, block_fid, comb_stencil)
            sampled_lid, sampled_data = sm.get_samples(block_data, comb_stencil)
            
            #sampled_fid = block_fid[comb_stencil > 0.5]
            sampled_fid= np.zeros_like(sampled_lid,dtype=np.int)
            sampled_locs = np.where(comb_stencil > 0.5)[0]
            #print(np.shape(sampled_locs))

            list_sampled_lid[bid]= sampled_lid
            list_sampled_data[bid]= sampled_data

            array_void_hist[bid,:] = void_hist
            array_ble[bid] = ble
            array_delta[bid] = delta
            
            # write out a vtp file
            # now use this stencil array to store the locations
            name='dm_density'
            #int_inds = np.where(stencil_new>0.5)
            #poly_data = vtk.vtkPolyData()
            Points = vtk.vtkPoints()
            #print('data type is:',sampled_data.dtype)
            if sampled_data.dtype=='float64':
                val_arr = vtk.vtkDoubleArray()
            elif sampled_data.dtype=='float32':
                val_arr = vtk.vtkFloatArray()
            #val_arr = vtk.vtkDoubleArray()
            #val_arr = vtk.vtkFloatArray()
            val_arr.SetNumberOfComponents(1)
            val_arr.SetName(name)

            
            bd_idx = np.unravel_index(bid,bd_dims)
            sampled_locs_xyz = np.unravel_index(sampled_locs,blk_dims)
            #print(np.multiply(bd_idx,blk_dims),np.shape(sampled_locs_xyz),sampled_locs_xyz)
            pt_locs_np = np.add(np.transpose(sampled_locs_xyz), np.multiply(bd_idx,blk_dims))
            #pt_locs_np = pt_locs_np.T
            pt_locs_np[:,[0,2]] = pt_locs_np[:,[2,0]]
            #print('here:',pt_locs_np)
            Points.SetData(VN.numpy_to_vtk(pt_locs_np))
                
            val_arr.SetArray(sampled_data,sampled_data.size,True)
            val_arr.array = sampled_data
            
            tot_points[bid] = sampled_data.size
            
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(Points)

            polydata.GetPointData().AddArray(val_arr)
            
            ## write the vtp file
            writer = vtk.vtkXMLPolyDataWriter();
            writer.SetFileName("vtu_outputs/sampled_"+sampling_method+'_pymp/'+"sampled_"+sampling_method+"_"+str(bid)+".vtp");
            if vtk.VTK_MAJOR_VERSION <= 5:
                writer.SetInput(polydata)
            else:
                writer.SetInputData(polydata)
            writer.Write()

    #print('list_sampled_lid:',list_sampled_lid)
    #print('\n')    

    filename = "vtu_outputs/sampled_"+sampling_method+"_pymp.vtm"

    file = open(filename, "w")
    top_string = '<?xml version="1.0"?> \n <VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor"> \n <vtkMultiBlockDataSet>'
    bottom_string = '\n </vtkMultiBlockDataSet> \n</VTKFile>'
    file.write(top_string)
    file_count = 0
    for bid in range(whichBlock,nob,freq):
        middle_string = '\n  <DataSet index="'+str(file_count)+'" file="sampled_'+sampling_method+'_pymp/sampled_'+sampling_method+'_'+str(bid)+'.vtp"/>'
        file.write(middle_string)
        file_count+=1
    file.write(bottom_string)
    file.close()

    ## store the samples and related information
    sampling_method_val = np.int(enum_dict[sampling_method])
    #print('sampling_method_val:',sampling_method_val)
    #out_folder = outPath
    #out_folder = 'sampled_output'
    if not os.path.exists(out_folder):
        print("Output path with sampled output does not exist! Creating..")
        os.makedirs(out_folder)
        #sys.exit()
    # ghist_params_list
    dump_with_pickle(ghist_params_list[0:2], out_folder+'/'+"ghist_params_list.pickle")
    dump_with_pickle(bm_paramter_list, out_folder+'/'+"bm_paramter_list.pickle")
    dump_with_pickle(list(list_sampled_lid), out_folder+'/'+"list_sampled_lid.pickle")
    dump_with_pickle(list(list_sampled_data), out_folder+'/'+"list_sampled_data.pickle")
    np.asarray(array_void_hist).tofile(out_folder+'/'+"array_void_hist.raw")
    np.asarray(array_ble).tofile(out_folder+'/'+"array_ble.raw")
    np.asarray(array_delta).tofile(out_folder+'/'+"array_delta.raw")
    #np.asarray(sampling_rate).tofile(out_folder+"sampling_rate.raw")
    #np.asarray(sampling_method_val).tofile(out_folder+"sampling_method_val.raw")
    np.save(out_folder+'/'+"sampling_rate.npy",np.asarray(sampling_rate))
    np.save(out_folder+'/'+"sampling_method_val.npy",np.asarray(sampling_method_val))

    zip_folder = out_folder+'_archive'+str(sampling_ratio)+'_'+sampling_method
    shutil.make_archive(zip_folder, 'zip', out_folder)
    print('\nSize of the compressed data:',os.path.getsize(zip_folder+'.zip')/1000000,' MB')
    print('Size of the original data:',os.path.getsize(infile)/1000000,' MB') #infile
    print('Effective Compression Ratio:',os.path.getsize(zip_folder+'.zip')/os.path.getsize(infile))

    print('Total points stored:',np.sum(tot_points))
    print('Time taken %.2f secs' %(time.time()-t0))

parser = argparse.ArgumentParser()

# if len(sys.argv) != 5:
#     parser.error("incorrect number of arguments")

parser.add_argument('--input', action="store", required=True,help="input file name")
parser.add_argument('--output', action="store", required=False,help="output folder name")
parser.add_argument('--percentage', action="store", required=False,help="what fraction of samples to keep")
parser.add_argument('--nbins', action="store", required=False,help="how many bins to use")
parser.add_argument('--blk_dims', action="store", required=False,help="block dimensions. Supporting cubing blocks for now. blk_dim --> [blk_dim,blk_dim,blk_dim] ")
parser.add_argument('--nthreads', action="store", required=False,help="how many threads to use")
parser.add_argument('--method', action="store", required=True,help="which sampling method to use. hist, grad, hist_grad, random, mixed")

args = parser.parse_args()

infile = getattr(args, 'input')
outPath = getattr(args, 'output')
sampling_ratio = getattr(args, 'percentage')
nthreads = getattr(args, 'nthreads')
nbins = getattr(args, 'nbins')
nblk_dims = getattr(args, 'blk_dims')

method = getattr(args, 'method')

if sampling_ratio==None:
    sampling_ratio = 0.01
else:   
    sampling_ratio = float(getattr(args, 'percentage'))

if method==None:
    method = 'hist'

if nthreads==None:
    nthreads = 4
else:
    nthreads = int(getattr(args, 'nthreads'))

if nblk_dims==None:
    nblk_dims = 10
else:
    nblk_dims = int(getattr(args, 'blk_dims'))

if outPath==None:
    outPath='_sampled_output_'
outPath = str(Path(outPath))

enum_dict = {'rand':0,'hist':1,'hist_grad':2,'hist_grad_rand':3, 'grad':4}

ghist_params_list = [16,16,nblk_dims,nblk_dims,nblk_dims]

run(infile, sampling_ratio,0.0, method,nthreads,outPath,ghist_params_list)

## python nyx_test_ayan_pymp.py --input=nyx_data_dark_matter_density.vti --method='hist' --percentage=0.005 --nthreads=6
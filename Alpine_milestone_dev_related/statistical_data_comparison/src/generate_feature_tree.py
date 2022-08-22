import matplotlib.pyplot as plt
import vtk
import numpy as np
import sys
import math
import os
import glob
import time
from vtk.util.numpy_support import *
import pandas as pd
from multiprocessing import Pool
from vtk.util import numpy_support
import pickle

def write_vtp(filename,data):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()

def ensureUtf(s):
    try:
        if type(s) == unicode:
            return s.encode('utf8', 'ignore')
    except: 
        return str(s)

def write_multiblock(filename,data):
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()
#################################
track_tstep = 170
initT = 75
endT = 408

all_bubble_info_file = '../out/segmented_features/all_bubble_info.csv'
pickle_file_path = '../out/segmented_features/'
feature_path = '../out/mfix_local_grid/'
full_cdb_path = '../bubble_all.cdb/'
in_tracked_feature_cdb_path = '../out/feature_tracking_cdb/'
######################################################################

## Load all bubble info to pandas
all_bubble_info = pd.read_csv(all_bubble_info_file)

## load tracking time step pickle file
inpicklefname = pickle_file_path + 'feature_values_' + str(track_tstep) + '.pickle'
ffname = ensureUtf(inpicklefname)
fname = open(ffname, 'rb')
data = pickle.load(fname, encoding='latin1')
data = np.asarray(data)
## get the feature ids which are tracked
feature_ids = data[:,1]

## create the multiblock
mb = vtk.vtkMultiBlockDataSet()
#mb.SetNumberOfBlocks(len(feature_ids))

# pts = vtk.vtkPoints()
# volume_arr = vtk.vtkFloatArray()
# volume_arr.SetName('bubble_volume')
# tstep_arr = vtk.vtkIntArray()
# tstep_arr.SetName('bubble_tstep')
# vel_arr = vtk.vtkFloatArray()
# vel_arr.SetName('bubble_velocity')

iter_val=0
for i in feature_ids:
	cdb_path = in_tracked_feature_cdb_path + 'bubble_tracked_' + str(track_tstep) + '_' + str(int(i)) + '.cdb'
	cdb_path = cdb_path + '/all_info.csv'
	#print (cdb_path)

	single_feature_track_info = pd.read_csv(cdb_path)  
	#print (single_feature_track_info)

	pdata = vtk.vtkPolyData()
	pts = vtk.vtkPoints()

	volume_arr = vtk.vtkFloatArray()
	volume_arr.SetName('bubble_volume')

	tstep_arr = vtk.vtkIntArray()
	tstep_arr.SetName('bubble_tstep')

	vel_arr = vtk.vtkFloatArray()
	vel_arr.SetName('bubble_velocity')	

	
	for index, row in single_feature_track_info.iterrows():
		tstep = row['time_step']
		fid = row['feature_id']
		velocity = row['rise_velocity']

		idx = np.where((all_bubble_info['time_step']==tstep) & (all_bubble_info['feature_id']==fid))
		df_selected = all_bubble_info.loc[idx]

		df_selected = df_selected.drop(df_selected.columns[[1,2,3,4,9]], axis=1)
		arr = (df_selected.to_numpy()[0])
		
		vel_arr.InsertNextTuple1(velocity)
		tstep_arr.InsertNextTuple1(arr[0])
		volume_arr.InsertNextTuple1(arr[1])
		pts.InsertNextPoint(arr[2],arr[3],arr[4])

	pline = vtk.vtkPolyLine()
	pline.GetPointIds().SetNumberOfIds(pts.GetNumberOfPoints());
	
	for ii in range(pts.GetNumberOfPoints()):
		pline.GetPointIds().SetId(ii,ii)

	cells = vtk.vtkCellArray()
	cells.InsertNextCell(pline)

	pdata.SetPoints(pts)
	pdata.SetLines(cells)
	pdata.GetPointData().AddArray(volume_arr)
	pdata.GetPointData().AddArray(vel_arr)
	pdata.GetPointData().AddArray(tstep_arr)

	mb.SetBlock(int(i),pdata)
	iter_val=iter_val+1


write_multiblock('/Users/sdutta/Desktop/out.vtm',mb)


		






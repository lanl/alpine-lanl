## Post-hoc feature tracking algorithm development.
## Algorithm developed for ECP Alpine Project.
## This code is delivered as part of Alpine 16-14 P6 activity.
## Please contact Soumya Dutta (sdutta@lanl.gov) for any questions.
## Unauthorized use of this code is prohibited.


## This script takes a list of vti files as inputs which are considered as the feature similarity field (generated in situ).
## This script processes those vti files and generates a pickle file for each time step which contains all the feature attributes
## for each time step and also generates a vtm file which contains the segmented features for each time step as unstructured 
## grid data. Finally, this script also generates a csv file which contains all the feature attributes.

####################################################################################################################################

import vtk
import numpy as np
import sys
import math
import os
import glob
import time
from vtk.util.numpy_support import *
import pandas
from multiprocessing import Pool
from vtk.util import numpy_support
from sklearn.decomposition import PCA
from numpy import linalg as LA
import pickle

#############################################

def ensureUtf(s):
    try:
        if type(s) == unicode:
            return s.encode('utf8', 'ignore')
    except: 
        return str(s)

def read_vti(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def write_vti(filename,data):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()
    
def write_vtu(filename,data):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()    
    
def write_multiblock(filename,data):
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()    
    
def compute_bubble_features(fname,confidence_th,size_threshold, tstep, outpath):
    
    feature_values=[]
        
    data = read_vti(fname)
    data.GetPointData().SetActiveScalars('feature_similarity')
    gbounds = data.GetBounds()
    
    thresholding = vtk.vtkThreshold()
    thresholding.ThresholdByUpper( confidence_th )
    thresholding.SetInputData(data)

    segmentation = vtk.vtkConnectivityFilter()
    segmentation.SetInputConnection(thresholding.GetOutputPort())
    segmentation.SetExtractionModeToAllRegions()
    segmentation.ColorRegionsOn()
    segmentation.Update()

    ug = segmentation.GetOutput()
    num_segments = segmentation.GetNumberOfExtractedRegions()
    
    ## compute volumes of each bubble: 
    ## volume = voxel count
    bubble_volumes = np.zeros(num_segments)
    for i in range(ug.GetPointData().GetArray('RegionId').GetNumberOfTuples()):
        regionId = int(ug.GetPointData().GetArray('RegionId').GetTuple(i)[0])
        bubble_volumes[regionId] = bubble_volumes[regionId]+1
        
    num_valid_features=0
    for i in range(num_segments):
        if  bubble_volumes[i] > size_threshold:
            num_valid_features=num_valid_features+1       
    
    find_max_topmost_xvals=[]
    idx=0
    ## Iterate over each feature and compute different attributes
    for i in range(num_segments):
        if  bubble_volumes[i] > size_threshold:
            #print ('processing bubble: '+ str(i))
            thresholding2 = vtk.vtkThreshold()
            thresholding2.SetInputData(ug)
            thresholding2.ThresholdBetween(i,i)
            thresholding2.Update()

            single_feature = thresholding2.GetOutput()
            
            ## compute aspect ratio
            #########################
            bounds = single_feature.GetBounds()
            aspect_ratio_3d = (bounds[5] - bounds[4])/(bounds[1] - bounds[0])
            
            ## get points to a numpy array
            feature_pts = numpy_support.vtk_to_numpy(single_feature.GetPoints().GetData())
            
            ## compute centroid
            #######################
            x_center = np.mean(feature_pts[:,0])
            y_center = np.mean(feature_pts[:,1])
            z_center = np.mean(feature_pts[:,2])

            # ## testing with max values for tracking: change back to centorid
            # x_center = np.max(feature_pts[:,0])
            # y_center = np.max(feature_pts[:,1])
            # z_center = np.max(feature_pts[:,2])
            
            ## record the max x-axis/vertical-axis values
            max_x = np.max(feature_pts[:,0])
            find_max_topmost_xvals.append([max_x,i,idx])
                        
            roundness = 0.0 ## not using roundness

            ## Append all feature values to a list
            feature_values.append([tstep,i,idx,aspect_ratio_3d,roundness,bubble_volumes[i],x_center,y_center,z_center,max_x])
            idx=idx+1
                    
    ## identify the indices of features which are actually topmost and not the actual desired bubbles
    find_max_topmost_xvals = np.asarray(find_max_topmost_xvals)
    top_indices=[]
    for i in range(len(find_max_topmost_xvals)):
        diff = np.abs(gbounds[1]-find_max_topmost_xvals[i,0])
        if diff < 0.00005:
            top_indices.append(int(find_max_topmost_xvals[i,1]))   

    ## filter out the top most undesired features and write the file out with bubble attributes
    temp_list = [item for item in feature_values if item[1] not in(top_indices)]

    ## Write the final filtered features into disks
    mb = vtk.vtkMultiBlockDataSet()
    mb.SetNumberOfBlocks(len(temp_list))
    final_filtered_feature_values=[]

    for j in range(len(temp_list)):
        fid = temp_list[j][1]
        thresholding2 = vtk.vtkThreshold()
        thresholding2.SetInputData(ug)
        thresholding2.ThresholdBetween(fid,fid)
        thresholding2.Update()
        single_feature = thresholding2.GetOutput()
        ## Add the block to the multiblock dataset
        mb.SetBlock(j,single_feature)
        final_filtered_feature_values.append(temp_list[j])

    file_name = outpath + 'segmented_' + str(tstep) + '.vtm'
    write_multiblock(file_name, mb)        

    return final_filtered_feature_values

#################################################################################

## Parameters
###############
init_time = 75
end_time = 100
## Folder that contains all the feature similarity fields (This fields will be generated in situ)
input_path = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_local_grid/'

## Folder that will contain the pickle files, vtm files and the csv file
outpath = '../out/segmented_features/'
outpath_csv = '../out/bubble_all.cdb/'
## similarity threshold for segmentation
confidence_th = 0.92
## feature size threshold. The value indicates number of voxels. 
## any feature below this threshold will be discarded.
size_threshold = 100

#################################################################################
start_time = time.time()

## Runs over all time steps, computes bubble feature attributes, stores segmend features and pickle files
for ii in range(init_time,end_time):
	print ('Processing time step: ' + str(ii))
	inpfname = input_path + 'slic_compare_' + str(ii) + '.vti'
	feature_values_ret = compute_bubble_features(inpfname,confidence_th,size_threshold, ii, outpath)
	outpicklefname = outpath + 'feature_values_' + str(ii) + '.pickle'
	pickle.dump(feature_values_ret, open( outpicklefname, "wb"))

print ('Generating the csv file..')
### stack all data and  generate csv file for visualization
outfname = outpath_csv + 'data.csv'
all_features = np.empty(9)

## Reload feature values from pickle file
for ii in range(init_time,end_time):
    inpicklefname = outpath + 'feature_values_' + str(ii) + '.pickle'
    ffname = ensureUtf(inpicklefname)
    fname = open(ffname, 'rb')
    data = pickle.load(fname, encoding='latin1')
    data = np.asarray(data)

    ## this is because the first time the array is empty and cannot use vstack
    if ii == init_time:
        all_features = data
    else:    
        all_features = np.vstack((all_features,data))

df = pandas.DataFrame(all_features, index=range(all_features.shape[0]), columns=range(all_features.shape[1]))

## Name the columns
df.rename(columns={0:'time_step'}, inplace=True)
df.rename(columns={1:'feature_id'}, inplace=True)
df.rename(columns={2:'sid'}, inplace=True)
df.rename(columns={3:'aspect_ratio'}, inplace=True)
df.rename(columns={4:'eigen_ratio'}, inplace=True)
df.rename(columns={5:'volume'}, inplace=True)
df.rename(columns={6:'x_center'}, inplace=True)
df.rename(columns={7:'y_center'}, inplace=True)
df.rename(columns={8:'z_center'}, inplace=True)
df.rename(columns={9:'x_max'}, inplace=True)

#### store all the info before dropping columns into a separate csv for generating tracking graph
out_csv_path = outpath_csv + 'all_bubble_info.csv'
df.to_csv(out_csv_path,index=False)

## drop some unnecessary columns
# 2 = serial id of bubble, 7 = y_center, 8 = z_center, 9 = x_max
df = df.drop(df.columns[[2, 4, 7, 8, 9]], axis=1)

all_fnames=[]
for i in range(len(all_features)):
    fname = 'images/bubble_' + str(int(all_features[i][1])) + '_' + str(int(all_features[i][0])) + '.png'
    all_fnames.append(fname)
    
## add a last column with filenames
df['FILE'] = all_fnames  

## write to disk
df.to_csv(outfname,index=False)
print ('done generating csv')

end_time = time.time()
print ('Total time taken: ' + str(end_time-start_time))



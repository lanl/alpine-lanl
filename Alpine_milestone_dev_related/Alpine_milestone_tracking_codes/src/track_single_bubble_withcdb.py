## This script tracks a single bubble foreward and backward from the current time step 
########################################################################################

import matplotlib.pyplot as plt
import vtk
import numpy as np
import sys
import math
import os
import glob
from vtk.util.numpy_support import *
import pandas as pd
from multiprocessing import Pool
from vtk.util import numpy_support
from sklearn.decomposition import PCA
from numpy import linalg as LA
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
import pickle
from sklearn import manifold
from numpy import linalg as LA
import time

################################################

def read_multiblock(filename):
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()
    
def write_multiblock(filename,data):
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()

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
    
def compute_3d_to_1d_map(x,y,z,dimx,dimy,dimz):

    return x + dimx*(y+dimy*z)    
    
def extract_target_feature(feature_volume,confidence_th,feature_id):
    
    feature_volume.GetPointData().SetActiveScalars('feature_similarity')
    thresholding = vtk.vtkThreshold()
    thresholding.ThresholdByUpper(confidence_th)
    thresholding.SetInputData(feature_volume)
    seg = vtk.vtkConnectivityFilter()
    seg.SetInputConnection(thresholding.GetOutputPort())
    seg.SetExtractionModeToAllRegions()
    seg.ColorRegionsOn()
    thresholding2 = vtk.vtkThreshold()
    thresholding2.SetInputConnection(seg.GetOutputPort())
    thresholding2.ThresholdBetween(feature_id,feature_id)
    thresholding2.Update()
    return thresholding2.GetOutput()

def get_x_center(target_feature):
    ## get points to a numpy array
    feature_pts = numpy_support.vtk_to_numpy(target_feature.GetPoints().GetData())
    return np.mean(feature_pts[:,0])

def get_x_max(target_feature):
    ## get points to a numpy array
    feature_pts = numpy_support.vtk_to_numpy(target_feature.GetPoints().GetData())
    return np.max(feature_pts[:,0])

def compute_1d_indices(single_feature,dims,gbounds):
    
    oneD_indices = [] 
    
    for i in range(single_feature.GetNumberOfPoints()):
        pts = single_feature.GetPoint(i)
        xval = int(((pts[0] - gbounds[0])/(gbounds[1]-gbounds[0]))*dims[0])
        yval = int(((pts[1] - gbounds[2])/(gbounds[3]-gbounds[2]))*dims[1])
        zval = int(((pts[2] - gbounds[4])/(gbounds[5]-gbounds[4]))*dims[2])
        
        if xval > dims[0]-1:
            xval = dims[0]-1
        
        if yval > dims[1]-1:
            yval = dims[1]-1
            
        if zval > dims[2]-1:
            zval = dims[2]-1    
        
        index = compute_3d_to_1d_map(xval,yval,zval,dims[0],dims[1],dims[2])
        oneD_indices.append(index)
        
        if index > dims[0]*dims[1]*dims[2] or index < 0:
            print (xval,yval,zval)
    
    if len(oneD_indices) != single_feature.GetNumberOfPoints():
        print ('mismatch in mumber of points!!')
        
    return oneD_indices

### dice Similarity function of 2 set of points
def dice_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set(x)) + len(set(y))
    return 2*intersection_cardinality/float(union_cardinality)

def find_match(data,target_feature,confidence_th,dims,size_threshold,gbounds,pickle_data,target_feature_x_center):
    
    num_features = data.GetNumberOfBlocks()
    matched = []
    
    for i in range(num_features):
        
        block = data.GetBlock(i)
        vtk_pts = block.GetPoints()
        num_feature_pts = vtk_pts.GetNumberOfPoints()

        if (num_feature_pts > size_threshold):
        
            oneD_indices = []
            for j in range(num_feature_pts):
                
                pts = vtk_pts.GetPoint(j)
                xval = int(((pts[0] - gbounds[0])/(gbounds[1]-gbounds[0]))*dims[0])
                yval = int(((pts[1] - gbounds[2])/(gbounds[3]-gbounds[2]))*dims[1])
                zval = int(((pts[2] - gbounds[4])/(gbounds[5]-gbounds[4]))*dims[2])

                if xval > dims[0]-1:
                    xval = dims[0]-1
                if yval > dims[1]-1:
                    yval = dims[1]-1
                if zval > dims[2]-1:
                    zval = dims[2]-1    

                index = compute_3d_to_1d_map(xval,yval,zval,dims[0],dims[1],dims[2])
                if index > dims[0]*dims[1]*dims[2] or index < 0:
                    print ('Index value out of bound!!')
                else:
                    oneD_indices.append(index)
            
            #intersection_cardinality = len(set.intersection(*[set(target_feature), set(oneD_indices)]))
            overlap = dice_similarity(target_feature,oneD_indices)
            
            ## If an overlap is found and if the overlapped object is in a higher position vertically
            #if overlap > 0 and pickle_data[i][6] > target_feature_x_center:
            
            matched_feature_vel = np.abs(pickle_data[i][6] - target_feature_x_center)
            
            if overlap > 0:    
                matched.append([pickle_data[i][0],pickle_data[i][1],overlap,oneD_indices,pickle_data[i][9],matched_feature_vel,pickle_data[i][6]])
       
    return matched

##############################################################

## Parameters ################
confidence_th = 0.92
size_threshold = 100
dims  = [128,16,128]
initT = 75
endT = 100

## Feature to track: User provided:

# feature_tstep = 75 #(in Paraview should subtract 75)
# feature_id = 1

feature_tstep = 75 #(in Paraview should subtract 75)
feature_id = 25

############################################################
############################################################
feature_path = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_local_grid/'
feature_fname = feature_path + 'slic_compare_' + str(feature_tstep) + '.vti'
print ('initial feature file to load: ' + feature_fname)

data_path = '../out/segmented_features/'
full_cdb_path = '../out/bubble_all.cdb/'
segmented_feature_volumes = '../out/segmented_volumes/'
out_tracked_feature_cdb_path = '../out/feature_tracking_cdb/'

start_time = time.time()

## Forward in time tracking
th = 0.004 ## important threshold controls the robustness

feature_volume = read_vti(feature_fname)
gbounds = feature_volume.GetBounds()
target_feature = extract_target_feature(feature_volume,confidence_th,feature_id)
target_feature_1d_pts_og = compute_1d_indices(target_feature,dims,gbounds)
target_feature_x_max_og = get_x_max(target_feature)
target_feature_x_center_og = get_x_center(target_feature)
##############################################################

#### forward_tracking #################################
target_feature_1d_pts = target_feature_1d_pts_og
feature_id_temp = feature_id
target_feature_x_max = target_feature_x_max_og
target_feature_x_center = target_feature_x_center_og

forward_tracking = []
for ii in range(feature_tstep+1,endT):
    ## load features to be tested
    inpfname = data_path + 'segmented_' + str(ii) + '.vtm'
    current_data = read_multiblock(inpfname)
    
    #load pickle file
    pickle_fname = data_path + 'feature_values_' + str(ii) + '.pickle'
    ffname = ensureUtf(pickle_fname)
    fname = open(ffname, 'rb')
    data = pickle.load(fname, encoding='latin1')
    data = np.asarray(data)
    
    matched = find_match(current_data,target_feature_1d_pts,confidence_th,dims,size_threshold,gbounds,data,target_feature_x_center)
           
    if (len(matched)>1):
        # case: multiple object overlapped: either feature split or continuation
        
        #print ('target_feature_x_center: '+str(target_feature_x_center))  
        print ('in forward')      
        for kk in range(len(matched)):
            print (ii,matched[kk][1],matched[kk][2],matched[kk][4])

        ## filter out the top most undesired features and write the file out with bubble attributes
        #temp_matched = [item for item in matched if item[4] > target_feature_x_center] 
        temp_matched = [item for item in matched if (np.abs(item[4] - target_feature_x_max) < th) \
                       or (item[4] > target_feature_x_max)] 
        matched = temp_matched
        
        ## in this case sort based on dice val so that the best match is selected to continue
        matched.sort(key = lambda matched: matched[2], reverse=True)
                
        ## update the target feature
        target_feature_1d_pts = matched[0][3]
        target_feature_x_max = matched[0][4]
        target_feature_x_center = matched[0][6]        
        forward_tracking.append([matched[0][0],matched[0][1],matched[0][2],matched[0][5]])
    
    elif len(matched) == 1:
        # filter out the top most undesired features and write the file out with bubble attributes
        #temp_matched = [item for item in matched if item[4] > target_feature_x_center]
        temp_matched = [item for item in matched if (np.abs(item[4] - target_feature_x_max) < th) \
                       or (item[4] > target_feature_x_max)] 
        matched = temp_matched
        
        if (len(matched)> 0):
                
            ## update the target feature
            target_feature_1d_pts = matched[0][3]
            target_feature_x_max = matched[0][4]
            target_feature_x_center = matched[0][6]
            forward_tracking.append([matched[0][0],matched[0][1],matched[0][2],matched[0][5]])
    else:
        #print ('No match found for foreward tracking')
        break
        
print ('done foreward tracking')         

###### backward_tracking ##############################
target_feature_1d_pts = target_feature_1d_pts_og
feature_id_temp = feature_id
target_feature_x_max = target_feature_x_max_og
target_feature_x_center = target_feature_x_center_og

backward_tracking = []
for ii in range(feature_tstep-1,initT,-1):
    
    ## load features to be tested
    inpfname = data_path + 'segmented_' + str(ii) + '.vtm'
    current_data = read_multiblock(inpfname)
    
    #load pickle file
    pickle_fname = data_path + 'feature_values_' + str(ii) + '.pickle'
    ffname = ensureUtf(pickle_fname)
    fname = open(ffname, 'rb')
    data = pickle.load(fname, encoding='latin1')
    data = np.asarray(data)
    
    matched = find_match(current_data,target_feature_1d_pts,confidence_th,dims,size_threshold,gbounds,data,target_feature_x_center)
    	    
    if (len(matched)>1):
        # case: multiple object overlapped: either feature split or continuation
        
        #print ('target_feature_x_center: '+str(target_feature_x_center)) 
        print ('in backward')        
        for kk in range(len(matched)):
            print (ii,matched[kk][1],matched[kk][2],matched[kk][4])
            
        ## filter out the top most undesired features and write the file out with bubble attributes
        #temp_matched = [item for item in matched if item[4] > target_feature_x_center] 
        temp_matched = [item for item in matched if (np.abs(item[4] - target_feature_x_max) < th) \
                       or (item[4] > target_feature_x_max)] 
        matched = temp_matched
        
        ## in this case sort based on dice val so that the best match is selected to continue
        matched.sort(key = lambda matched: matched[2], reverse=True)
                        
        ## update the target feature
        target_feature_1d_pts = matched[0][3]
        target_feature_x_max = matched[0][4]
        target_feature_x_center = matched[0][6]
        backward_tracking.append([matched[0][0],matched[0][1],matched[0][2],matched[0][5]])
    
    elif len(matched) == 1:
        # filter out the top most undesired features and write the file out with bubble attributes
        #temp_matched = [item for item in matched if item[4] > target_feature_x_center]
        temp_matched = [item for item in matched if (np.abs(item[4] - target_feature_x_max) < th) \
                       or (item[4] > target_feature_x_max)] 
        matched = temp_matched
        
        if (len(matched)> 0):
            ## case: single object overlapped: feature continuation
            #print ('Matcheddd feature: tstep: ' + str(matched[0][0]) + ' ' + str(matched[0][0]-75) +  ' fid: ' +  str(matched[0][1]))
                
            ## update the target feature
            target_feature_1d_pts = matched[0][3]
            target_feature_x_max = matched[0][4]
            target_feature_x_center = matched[0][6]
            backward_tracking.append([matched[0][0],matched[0][1],matched[0][2],matched[0][5]])
    else:
        #print ('No match found for foreward tracking')
        break
        
print ('done backward tracking')     

end_time = time.time()

print ('tracking time:  ' + str(end_time-start_time))

###########################################################################
start_cdb_time = time.time()

## load the complete database and filter the target specific cinemadatabase
## first combine all tracking results
all_tracked_results = backward_tracking + forward_tracking

## get the avg velocity which will be used for the starting/pivot feature's velocity
all_velocities = np.asarray(all_tracked_results)[:,3]
avg_vel = np.mean(all_velocities)

all_diceindices = np.asarray(all_tracked_results)[:,2]
avg_dice = np.mean(all_diceindices)

all_tracked_results.append([feature_tstep,feature_id,avg_dice,avg_vel])

## sort the results by time step number
all_tracked_results.sort(key = lambda all_tracked_results: all_tracked_results[0]) 

##load csv file in dataframe
fname = full_cdb_path+'data.csv'
df = pd.read_csv(fname)

filtered_data=[]
for i in range(len(all_tracked_results)):
    data = df[(df['time_step']==all_tracked_results[i][0]) & (df['feature_id']==all_tracked_results[i][1])]
    data = data.to_numpy()
    data_list=list(data)    
    filtered_data.append(data_list)
    
all_tracked_results = np.asarray(all_tracked_results)
dice_vals = all_tracked_results[:,2]
feature_velocities = all_tracked_results[:,3]

## strip off one dimension, no changes to the actual data
final_filtered_data_tmp=[]
for i in range(np.shape(filtered_data)[0]):
    final_filtered_data_tmp.append(filtered_data[i][0])
 
final_filtered_data = np.asarray(final_filtered_data_tmp)

dice_vals = np.reshape(dice_vals, (len(dice_vals), 1))
feature_velocities = np.reshape(feature_velocities, (len(feature_velocities), 1))

final_filtered_data = np.append(final_filtered_data, dice_vals, axis=1)
final_filtered_data = np.append(final_filtered_data, feature_velocities, axis=1)

## clean the old result: this cleans the folder
os.system('rm -rf ' + out_tracked_feature_cdb_path + '*')

# mode for the folder 
mode = 0o777
out_feature_cdb_fname = out_tracked_feature_cdb_path + 'bubble_tracked_' + str(feature_tstep) + '_' + str(feature_id) +  '.cdb'
os.mkdir(out_feature_cdb_fname, mode)

images_path = out_feature_cdb_fname + '/images'
os.mkdir(images_path, mode)

df = pd.DataFrame(final_filtered_data, index=range(final_filtered_data.shape[0]),
                          columns=range(final_filtered_data.shape[1]))

## Name the columns
df.rename(columns={0:'time_step'}, inplace=True)
df.rename(columns={1:'feature_id'}, inplace=True)
df.rename(columns={2:'aspect_ratio'}, inplace=True)
df.rename(columns={3:'volume'}, inplace=True)
df.rename(columns={4:'x_center'}, inplace=True)
df.rename(columns={6:'dice_index'}, inplace=True)
df.rename(columns={7:'rise_velocity'}, inplace=True)
df.rename(columns={5:'FILE'}, inplace=True)

# reorder the dataframe before storing: This is after observing the relationships
df = df[['feature_id','time_step','volume','x_center','rise_velocity','aspect_ratio','dice_index','FILE']]

## store into csv file
out_csv_path = out_feature_cdb_fname + '/data.csv'
df.to_csv(out_csv_path,index=False)

###copy the images for this feature to cdb folder
for i in range(np.shape(final_filtered_data)[0]):
    ffname = full_cdb_path + 'images/bubble_' + str(int(final_filtered_data[i][1])) \
    + '_' + str(int(final_filtered_data[i][0])) + '.png'
    
    copy_command = 'cp ' + ffname + ' ' + out_feature_cdb_fname + '/images'
    os.system(copy_command)

###copy the feature volumes for this feature to cdb folder
for i in range(np.shape(final_filtered_data)[0]):
    ffname = segmented_feature_volumes + 'segmented_feature_' + str(int(final_filtered_data[i][1])) \
    + '_' + str(int(final_filtered_data[i][0])) + '.vti'
    copy_command = 'cp ' + ffname + ' ' + out_feature_cdb_fname + '/images'
    os.system(copy_command)
    
end_cdb_time = time.time()
print ('cdb creation time:  ' + str(end_cdb_time-start_cdb_time))

print ('done tracking and cdb is generated at: ' + out_feature_cdb_fname)











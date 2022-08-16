import vtk
import numpy as np
import sys
import math
import os
import glob
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import distance

## Isabel #######################################################################

percentage = 7 # 1,3,5
samp_type = 'pmi' # pmi

recon_type = 'linear' # nearest

raw_file1 = '../../Data/Isabel_vti/isabel_precip_25.vti'
raw_file2 = '../../Data/Isabel_vti/isabel_qgraup_25.vti'

raw_var1 = 'Precipitation'
raw_var2 = 'QGraup'

sampled_field1 = '../../output/joint_' + samp_type + '_recon_isabel_' + raw_var1 + '_' + recon_type + '_' + str(percentage) + '.vti'
sampled_field2 = '../../output/joint_' + samp_type + '_recon_isabel_' + raw_var2 + '_' + recon_type + '_' + str(percentage) + '.vti'

#VOI = [100,170,100,160,0,49] ## pressure+velocity
VOI = [100,150,100,140,0,25] ## precip+qgraup
#VOI = [0,249,0,249,0,49]

##################################################################################

## asteroid
#VOI = [0,100,200,299,100,210] # for tev
#VOI = [130,230,100,200,50,250]; # for v02
#VOI = [10,100,220,299,120,180]; # for v03

##################################################################################

def read_vti(filename):
	reader = vtk.vtkXMLImageDataReader()
	reader.SetFileName(filename)
	reader.Update()
	return reader.GetOutput()

def get_voi(raw_data,VOI):
	extractVOI = vtk.vtkExtractVOI()
	extractVOI.SetInputData(raw_data)
	extractVOI.SetVOI(VOI[0],VOI[1],VOI[2],VOI[3],VOI[4],VOI[5])
	extractVOI.Update()
	return extractVOI.GetOutput()

def compute_SNR(raw_vals, recon_vals):
	numPts = len(raw_vals)
	mean_raw = np.mean(raw_vals)
	mean_sampled = np.mean(recon_vals)
	stdev_raw = np.std(raw_vals)
	stdev_sampled = np.std(recon_vals)
	error = np.abs(raw_vals-recon_vals)
	mean_error = np.mean(error)
	stdev_error = np.std(error)
	return 20*math.log10(stdev_raw/stdev_error)

def compute_MI(raw_vals, recon_vals, numBins):
	numSamples = np.shape(raw_vals)[0]

	hist1 = np.histogram(raw_vals,bins=numBins)
	hist2 = np.histogram(recon_vals,bins=numBins)
	hist2D = np.histogram2d(raw_vals,recon_vals,bins=numBins)

	Array1 = hist1[0]
	Array2 = hist2[0]
	ArrayComb = hist2D[0]

	prob_of_x=0.0
	prob_of_y=0.0
	mi=0.0
	for i in range(0,numBins):
	    for j in range(0,numBins):
	    	if Array1[i] > 0 and Array2[j] > 0 and ArrayComb[i][j] > 0:

	    		prob_of_x = float(Array1[i]) / numSamples
	    		prob_of_y = float(Array2[j]) / numSamples
	    		prob_of_xy = float(ArrayComb[i][j]) / numSamples
	    		mi = mi + prob_of_xy*np.log2(prob_of_xy/(prob_of_x*prob_of_y))

	return mi    		

###################################################################################

## load raw data
raw_data1 = read_vti(raw_file1)
raw_data2 = read_vti(raw_file2)

recon_data1 = read_vti(sampled_field1)
recon_data2 = read_vti(sampled_field2)

## extract raw VOI for all the vars
raw_voi1 = get_voi(raw_data1,VOI)
raw_voi2 = get_voi(raw_data2,VOI)
recon_voi_var1 = get_voi(recon_data1,VOI)
recon_voi_var2 = get_voi(recon_data2,VOI)

## get dims of VOI
numPts = raw_voi1.GetPointData().GetArray(raw_var1).GetNumberOfTuples()

raw_vals1 = np.zeros(numPts)
raw_vals2 = np.zeros(numPts)

recon_vals1 = np.zeros(numPts)
recon_vals2 = np.zeros(numPts)

for i in range(numPts):
    raw_vals1[i] = raw_voi1.GetPointData().GetArray(raw_var1).GetTuple1(i)
    raw_vals2[i] = raw_voi2.GetPointData().GetArray(raw_var2).GetTuple1(i)

    recon_vals1[i] = recon_voi_var1.GetPointData().GetArray(raw_var1).GetTuple1(i)
    recon_vals2[i] = recon_voi_var2.GetPointData().GetArray(raw_var2).GetTuple1(i)

#########################################################################################
## compute the Pearson's correlation for each variable
corr = stats.pearsonr(raw_vals1, recon_vals1)
print 'Self Pearson\'s correlation for ' + raw_var1 + ' is: ' +  str(corr[0])

corr = stats.pearsonr(raw_vals2, recon_vals2)
print 'Self Pearson\'s correlation for ' + raw_var2 + ' is: ' +  str(corr[0])

# corr = stats.pearsonr(raw_vals1, raw_vals2)
# print 'Joint Pearson\'s correlation for raw data is: ' +  str(corr[0])

# corr = stats.pearsonr(recon_vals1, recon_vals2)
# print 'Joint Pearson\'s correlation for reconstruction is: ' +  str(corr[0])

#########################################################################################
## compute the Pearson's correlation for each variable
corr = distance.correlation(raw_vals1, recon_vals1)
print 'Self Distance correlation for ' + raw_var1 + ' is: ' +  str(corr)

corr = distance.correlation(raw_vals2, recon_vals2)
print 'Self Distance\'s correlation for ' + raw_var2 + ' is: ' +  str(corr)

# corr = distance.correlation(raw_vals1, raw_vals2)
# print 'Joint Distance\'s correlation for raw data is: ' +  str(corr)

# corr = distance.correlation(recon_vals1, recon_vals2)
# print 'Joint Distance\'s correlation for reconstruction is: ' +  str(corr)


#########################################################################################
## compute the MI for each variable
numBins = 128
mi = compute_MI(raw_vals1, raw_vals2, numBins)
print 'MI for raw data is: ' + str(mi)

mi = compute_MI(recon_vals1, recon_vals2, numBins)
print 'MI for recon data is: ' + str(mi)


############################################################
## compute SNR for each variable
snr = compute_SNR(raw_vals1, recon_vals1)
print 'SNR for ' + raw_var1 + ' is: ' +  str(snr)

snr = compute_SNR(raw_vals2, recon_vals2)
print 'SNR for ' + raw_var2 + ' is: ' +  str(snr)

# ##########################################################################
# ## Plot scatter plots

# # Create 2x1 sub plots
# gs = gridspec.GridSpec(2, 1)
# dot_size = 0.5
# color = 'black'

# plt.figure(figsize=(16, 8))

# ax = plt.subplot(gs[0, 0])
# plt.scatter(raw_vals1, raw_vals2, s=dot_size, c=color)
# ax.set_xlabel(raw_var1)
# ax.set_ylabel(raw_var2)

# ax = plt.subplot(gs[1, 0])
# plt.scatter(recon_vals1, recon_vals2, s=dot_size, c=color)
# ax.set_xlabel(recon_var1)
# ax.set_ylabel(recon_var2)

# plt.tight_layout()
# plt.show()
import vtk
import numpy as np
import sys
import math
import os
import glob
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

### Isabel #######################################################################
# raw_file = '../../data/Isabel_pressure_velocity_qvapor.vti'
# raw_var1 = 'Pressure'
# raw_var2 = 'Velocity'
# raw_var3 = 'QVapor'

# sampled_field1 = '../../analysis/recon_data/isabel_recon_Pressure_linear.vti'
# sampled_field2 = '../../analysis/recon_data/isabel_recon_Velocity_linear.vti'
# sampled_field3 = '../../analysis/recon_data/isabel_recon_QVapor_linear.vti'
# recon_var = 'ImageScalars'
# recon_var1 = 'Pressure'
# recon_var2 = 'Velocity'
# recon_var3 = 'QVapor'

# VOI = [115,160,100,150,0,49]

##################################################################################

### Asteroid #####################################################################
raw_file = '../../data/asteroid_28649.vti'
raw_var1 = 'tev'
raw_var2 = 'v02'
raw_var3 = 'v03'

sampled_field1 = '../../analysis/recon_data/asteroid_recon_tev_linear.vti'
sampled_field2 = '../../analysis/recon_data/asteroid_recon_v02_linear.vti'
sampled_field3 = '../../analysis/recon_data/asteroid_recon_v03_linear.vti'
recon_var = 'ImageScalars'
recon_var1 = 'tev'
recon_var2 = 'v02'
recon_var3 = 'v03'

#VOI = [0,100,200,299,100,210] # for tev
#VOI = [130,230,100,200,50,250]; # for v02
VOI = [10,100,220,299,120,180]; # for v03

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


###################################################################################

## load raw data
raw_data = read_vti(raw_file)
recon_data1 = read_vti(sampled_field1)
recon_data2 = read_vti(sampled_field2)
recon_data3 = read_vti(sampled_field3)

## extract raw VOI for all the vars
raw_voi = get_voi(raw_data,VOI)
recon_voi_var1 = get_voi(recon_data1,VOI)
recon_voi_var2 = get_voi(recon_data2,VOI)
recon_voi_var3 = get_voi(recon_data3,VOI)

## get dims of VOI
numPts = raw_voi.GetPointData().GetArray(raw_var1).GetNumberOfTuples()

raw_vals1 = np.zeros(numPts)
raw_vals2 = np.zeros(numPts)
raw_vals3 = np.zeros(numPts)

recon_vals1 = np.zeros(numPts)
recon_vals2 = np.zeros(numPts)
recon_vals3 = np.zeros(numPts)

for i in range(numPts):
    raw_vals1[i] = raw_voi.GetPointData().GetArray(raw_var1).GetTuple1(i)
    raw_vals2[i] = raw_voi.GetPointData().GetArray(raw_var2).GetTuple1(i)
    raw_vals3[i] = raw_voi.GetPointData().GetArray(raw_var3).GetTuple1(i)

    recon_vals1[i] = recon_voi_var1.GetPointData().GetArray(recon_var).GetTuple1(i)
    recon_vals2[i] = recon_voi_var2.GetPointData().GetArray(recon_var).GetTuple1(i)
    recon_vals3[i] = recon_voi_var3.GetPointData().GetArray(recon_var).GetTuple1(i)

#########################################################################################
## compute the Pearson's correlation for each variable
corr = stats.pearsonr(raw_vals1, recon_vals1)
print 'Pearson\'s correlation for ' + raw_var1 + ' is: ' +  str(corr[0])

corr = stats.pearsonr(raw_vals2, recon_vals2)
print 'Pearson\'s correlation for ' + raw_var2 + ' is: ' +  str(corr[0])

corr = stats.pearsonr(raw_vals3, recon_vals3)
print 'Pearson\'s correlation for ' + raw_var3 + ' is: ' +  str(corr[0])


############################################################
## compute SNR for each variable
snr = compute_SNR(raw_vals1, recon_vals1)
print 'SNR for ' + raw_var1 + ' is: ' +  str(snr)

snr = compute_SNR(raw_vals2, recon_vals2)
print 'SNR for ' + raw_var2 + ' is: ' +  str(snr)

snr = compute_SNR(raw_vals3, recon_vals3)
print 'SNR for ' + raw_var3 + ' is: ' +  str(snr)


#########################################################################################
## Plot scatter plots

# Create 2x3 sub plots
gs = gridspec.GridSpec(2, 3)
#fig, axes = plt.subplots(nrows=2, ncols=3)
dot_size = 0.5
color = 'black'

plt.figure(figsize=(16, 8))

ax = plt.subplot(gs[0, 0])
plt.scatter(raw_vals1, raw_vals2, s=dot_size, c=color)
ax.set_xlabel(raw_var1)
ax.set_ylabel(raw_var2)

ax = plt.subplot(gs[0, 1])
plt.scatter(raw_vals1, raw_vals3, s=dot_size, c=color)
ax.set_xlabel(raw_var1)
ax.set_ylabel(raw_var3)

ax = plt.subplot(gs[0, 2])
plt.scatter(raw_vals2, raw_vals3, s=dot_size, c=color)
ax.set_xlabel(raw_var2)
ax.set_ylabel(raw_var3)

ax = plt.subplot(gs[1, 0])
plt.scatter(recon_vals1, recon_vals2, s=dot_size, c=color)
ax.set_xlabel(recon_var1)
ax.set_ylabel(recon_var2)

ax = plt.subplot(gs[1, 1])
plt.scatter(recon_vals1, recon_vals3, s=dot_size, c=color)
ax.set_xlabel(recon_var1)
ax.set_ylabel(recon_var3)

ax = plt.subplot(gs[1, 2])
plt.scatter(recon_vals2, recon_vals3, s=dot_size, c=color)
ax.set_xlabel(recon_var2)
ax.set_ylabel(recon_var3)

plt.tight_layout()
plt.show()


import vtk
import numpy as np
import sys
import math
import os
import glob
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import dcor

from vtk.util import numpy_support

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', action="store", required=True, help="name of the data set")
args = parser.parse_args()
data = getattr(args, 'dataset')


## Isabel #######################################################################
if data=='isabel':
	## Isabel #######################################################################
	raw_file1 = '../Data/Isabel_vti/isabel_p_25.vti'
	raw_file2 = '../Data/Isabel_vti/isabel_vel_25.vti'
	#raw_file2 = '../Data/Isabel_vti/isabel_qva_25.vti'
	varname1 = 'Pressure'
	varname2 = 'Velocity' 

	### P and VEL
	## pmi
	sampled_field1 = '../output/recon_linear/pmi/P_VEL/joint_pmi_recon_isabel_Pressure_linear_5.vti'
	sampled_field2 = '../output/recon_linear/pmi/P_VEL/joint_pmi_recon_isabel_Velocity_linear_5.vti'
	## random
	rand_sampled_field1 = '../output/recon_linear/random/P_VEL/joint_random_recon_isabel_Pressure_linear_5.vti'
	rand_sampled_field2 = '../output/recon_linear/random/P_VEL/joint_random_recon_isabel_Velocity_linear_5.vti'

	outfname = 'Isabel_P_VEL.png'

	# #### P and QVA
	# ## pmi
	# sampled_field1 = '../output/recon_linear/pmi/P_QVA/joint_pmi_recon_isabel_Pressure_linear_5.vti'
	# sampled_field2 = '../output/recon_linear/pmi/P_QVA/joint_pmi_recon_isabel_QVapor_linear_5.vti'
	# ## random
	# rand_sampled_field1 = '../output/recon_linear/random/P_QVA/joint_random_recon_isabel_Pressure_linear_5.vti'
	# rand_sampled_field2 = '../output/recon_linear/random/P_QVA/joint_random_recon_isabel_QVapor_linear_5.vti'

	#outfname = 'Isabel_P_QVA.png'

	VOI = [100,160,100,160,0,49] ## not in use now not in paper
	#VOI = [100,180,75,170,0,30] # works for QVA and P in paper
	#VOI = [0,249,0,249,0,49] # works for QVA and P

	dot_size = 0.2
	color = 'black'
	alpha=0.2

# #################################################################################

# ### Asteroid #####################################################################
if data=='asteroid':
	
	raw_file1 = '../Data/Asteroid/tev.vti'
	raw_file2 = '../Data/Asteroid/v02.vti'
	varname1 = 'tev'
	varname2 = 'v02' 

	## pmi
	sampled_field1 = '../output/recon_linear/pmi/asteroid/joint_pmi_recon_asteroid_tev_linear_5.vti'
	sampled_field2 = '../output/recon_linear/pmi/asteroid/joint_pmi_recon_asteroid_v02_linear_5.vti'

	## random
	rand_sampled_field1 = '../output/recon_linear/random/asteroid/joint_random_recon_asteroid_tev_linear_5.vti'
	rand_sampled_field2 = '../output/recon_linear/random/asteroid/joint_random_recon_asteroid_v02_linear_5.vti'

	outfname = 'Asteroid_tev_v02.png'

	#VOI = [130,210,60,200,50,260] # for tev and v02 works
 	#VOI = [130,230,60,220,0,299] # for tev and v02 works  and used in paper
	VOI = [0,299,0,299,0,299] # for tev and v02 

	dot_size = 0.1
	color = 'black'
	alpha=0.05

#################################################################################

# ### Combustion #####################################################################
if data=='combustion':
	
	raw_file1 = '../Data/Combustion/combustion_mixfrac.vti'
	raw_file2 = '../Data/Combustion/combustion_Y_OH.vti'
	varname1 = 'mixfrac'
	varname2 = 'Y_OH' 

	## pmi
	sampled_field1 = '../output/recon_linear/pmi/combustion/joint_pmi_recon_combustion_mixfrac_linear_5.vti'
	sampled_field2 = '../output/recon_linear/pmi/combustion/joint_pmi_recon_combustion_Y_OH_linear_5.vti'

	## random
	rand_sampled_field1 = '../output/recon_linear/random/combustion/joint_random_recon_combustion_mixfrac_linear_5.vti'
	rand_sampled_field2 = '../output/recon_linear/random/combustion/joint_random_recon_combustion_Y_OH_linear_5.vti'

	outfname = 'Compustion_mixfrac_y_oh.png'

	VOI = [0,239,110,250,0,59] # for mixfrac and Y_OH in paper
	#VOI = [0,239,0,359,0,59] # for mixfrac and Y_OH

	dot_size = 0.1
	color = 'black'
	alpha=0.1

##################################################################################

def read_vti(filename):
	reader = vtk.vtkXMLImageDataReader()
	reader.SetFileName(filename)
	reader.Update()
	return reader.GetOutput()

def read_vtp(filename):
	reader = vtk.vtkXMLPolyeDataReader()
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
raw_data1 = read_vti(raw_file1)
raw_data2 = read_vti(raw_file2)

recon_data1 = read_vti(sampled_field1)
recon_data2 = read_vti(sampled_field2)

rand_recon_data1 = read_vti(rand_sampled_field1)
rand_recon_data2 = read_vti(rand_sampled_field2)

## extract raw VOI for all the vars
raw_voi1 = get_voi(raw_data1,VOI)
raw_voi2 = get_voi(raw_data2,VOI)
recon_voi1 = get_voi(recon_data1,VOI)
recon_voi2 = get_voi(recon_data2,VOI)

rand_recon_voi1 = get_voi(rand_recon_data1,VOI)
rand_recon_voi2 = get_voi(rand_recon_data2,VOI)

print 'loading done'

# ## get lenghth of VOI
# numPts = raw_voi1.GetPointData().GetArray(0).GetNumberOfTuples()

# # raw_vals1 = np.zeros(numPts)
# # raw_vals2 = np.zeros(numPts)
# # recon_vals1 = np.zeros(numPts)
# # recon_vals2 = np.zeros(numPts)
# # rand_recon_vals1 = np.zeros(numPts)
# # rand_recon_vals2 = np.zeros(numPts)

# raw_vals1 = []
# raw_vals2 = []
# recon_vals1 = []
# recon_vals2 = []
# rand_recon_vals1 = []
# rand_recon_vals2 = []
# counter=0

# for i in range(numPts):

#     d1 = recon_voi1.GetPointData().GetArray(0).GetTuple1(i)
#     d2 = recon_voi2.GetPointData().GetArray(0).GetTuple1(i)

#     if not (math.isnan(d1)) or not(math.isnan(d2)):

#     	# recon_vals1[i] = d1	
#     	# recon_vals2[i] = d2

#     	recon_vals1.append(d1)
#     	recon_vals2.append(d2)

#     	# rand_recon_vals1[i] = rand_recon_voi1.GetPointData().GetArray(0).GetTuple1(i)
#     	# rand_recon_vals2[i] = rand_recon_voi2.GetPointData().GetArray(0).GetTuple1(i)

#     	rand_recon_vals1.append(rand_recon_voi1.GetPointData().GetArray(0).GetTuple1(i))
#     	rand_recon_vals1.append(rand_recon_voi2.GetPointData().GetArray(0).GetTuple1(i))

#     	# raw_vals1[i] = raw_voi1.GetPointData().GetArray(0).GetTuple1(i)
#     	# raw_vals2[i] = raw_voi2.GetPointData().GetArray(0).GetTuple1(i)

#     	raw_vals1.append(raw_voi1.GetPointData().GetArray(0).GetTuple1(i))
#     	raw_vals2.append(raw_voi2.GetPointData().GetArray(0).GetTuple1(i))

#     else:
#     	counter=counter+1	


# print 'counter ' + str(counter)

# raw_vals1 = np.asarray(raw_vals1)
# raw_vals2 = np.asarray(raw_vals2)
# recon_vals1 = np.asarray(recon_vals1)
# recon_vals2 = np.asarray(recon_vals2)
# rand_recon_vals1 = np.asarray(rand_recon_vals1)
# rand_recon_vals2 = np.asarray(rand_recon_vals2)



## easily get the data without loop    
temp = raw_voi1.GetPointData().GetArray(varname1)
raw_vals1 = numpy_support.vtk_to_numpy(temp)

temp = raw_voi2.GetPointData().GetArray(varname2)
raw_vals2 = numpy_support.vtk_to_numpy(temp)

temp = recon_voi1.GetPointData().GetArray(varname1)
recon_vals1 = numpy_support.vtk_to_numpy(temp)
recon_vals1 = np.where(np.isnan(recon_vals1),0,recon_vals1)

temp = recon_voi2.GetPointData().GetArray(varname2)
recon_vals2 = numpy_support.vtk_to_numpy(temp)
recon_vals2 = np.where(np.isnan(recon_vals2),1,recon_vals2)


temp = rand_recon_voi1.GetPointData().GetArray(varname1)
rand_recon_vals1 = numpy_support.vtk_to_numpy(temp)

temp = rand_recon_voi2.GetPointData().GetArray(varname2)
rand_recon_vals2 = numpy_support.vtk_to_numpy(temp)


# pmi
########################################################################################
## compute the Pearson's correlation for each variable
corr = stats.pearsonr(raw_vals1, raw_vals2)
print 'raw correlation is: ' +  str(corr[0])

corr = stats.pearsonr(recon_vals1, recon_vals2)
print 'pmi recon correlation is: ' +  str(corr[0])

corr = stats.pearsonr(rand_recon_vals1, rand_recon_vals2)
print 'random recon correlation is: ' +  str(corr[0])

corr = dcor.distance_correlation(raw_vals1, raw_vals2)
print 'raw dcorr is: ' +  str(corr)

corr = dcor.distance_correlation(recon_vals1, recon_vals2)
print 'pmi recon dcorr is: ' +  str(corr)

corr = dcor.distance_correlation(rand_recon_vals1, rand_recon_vals2)
print 'random recon dcorr is: ' +  str(corr)

############################################################
## compute SNR for each variable
snr = compute_SNR(raw_vals1, recon_vals1)
print 'pmi var1 SNR: ' +  str(snr)

snr = compute_SNR(raw_vals1, rand_recon_vals1)
print 'rand var1 SNR: ' +  str(snr)

snr = compute_SNR(raw_vals2, recon_vals2)
print 'pmi var2 SNR: ' +  str(snr)

snr = compute_SNR(raw_vals2, rand_recon_vals2)
print 'rand var2 SNR: ' +  str(snr)


#########################################################################################
## Plot scatter plots

# Create 2x3 sub plots
gs = gridspec.GridSpec(1, 3)

plt.figure(figsize=(16, 4))

ax = plt.subplot(gs[0, 0])
plt.scatter(raw_vals1, raw_vals2, s=dot_size, c=color, alpha=alpha)
plt.title('Scatter plot of raw data')
ax.set_xlabel(varname1)
ax.set_ylabel(varname2)

ax = plt.subplot(gs[0, 1])
plt.scatter(recon_vals1, recon_vals2, s=dot_size, c=color, alpha=alpha)
plt.title('Scatter plot using PMI-based method')
ax.set_xlabel(varname1)
ax.set_ylabel(varname2)

ax = plt.subplot(gs[0, 2])
plt.scatter(rand_recon_vals1, rand_recon_vals2, s=dot_size, c=color, alpha=alpha)
plt.title('Scatter plot using random sampling method')
ax.set_xlabel(varname1)
ax.set_ylabel(varname2)

plt.tight_layout()
fname = '../Analysis/scatter_plots/' + outfname
print 'writing out ' + fname
plt.savefig(fname, dpi=300)
#plt.show()
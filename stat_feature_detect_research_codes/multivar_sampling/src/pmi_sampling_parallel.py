import sys
import subprocess
import re
import os
import time
import numpy as np
import cmath
import operator
import random
from operator import itemgetter, attrgetter
import vtk

from mpi4py import MPI

## import MI measures
from mi_functions import *

###isabel
file1 = '../Data/Isabel_vti/isabel_p_25.vti'
file2 = '../Data/Isabel_vti/isabel_vel_25.vti'
#file2 = '../Data/Isabel_vti/isabel_qva_25.vti'
# file1 = '../Data/Isabel_vti/isabel_precip_25.vti'
# file2 = '../Data/Isabel_vti/isabel_qgraup_25.vti'
arrayName1 = 'Pressure'
arrayName2 = 'Velocity'
data_set = 'isabel'
samp_type = 'pmi' # random, pmi
pmi_power = 5 ## P & Vel = 5, P & QVA = 7

# ## load vti data
R1 = vtk.vtkXMLImageDataReader()
R1.SetFileName(file1)
R1.Update()
dataArray1 = R1.GetOutput().GetPointData().GetArray(arrayName1)

R2 = vtk.vtkXMLImageDataReader()
R2.SetFileName(file2)
R2.Update()
dataArray2 = R2.GetOutput().GetPointData().GetArray(arrayName2)

var1=np.zeros(dataArray1.GetNumberOfTuples()) 
var2=np.zeros(dataArray2.GetNumberOfTuples()) 

for i in range(dataArray1.GetNumberOfTuples()):
    var1[i] = dataArray1.GetTuple1(i)
    var2[i] = dataArray2.GetTuple1(i)

dims = R1.GetOutput().GetDimensions()

min_var1 = np.min(var1)
max_var1 = np.max(var1)

min_var2 = np.min(var2)
max_var2 = np.max(var2)

print R1.GetOutput().GetSpacing()
print R1.GetOutput().GetOrigin()
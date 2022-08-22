import matplotlib.pyplot as plt
import vtk
import numpy as np
import sys
import math
import os
import glob
from vtk.util.numpy_support import *
import pandas
from multiprocessing import Pool

path = '/home/sdutta/Desktop/insitu_data_12500_out'
fname = path + '/*.vtp'

outfile = path + '/combined.vtp'

## combine a few vtp files into a single file
def read_vtp(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def write_vtp(filename,data):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Update()
    
pdata = vtk.vtkPolyData()
pts = vtk.vtkPoints()
velocity = vtk.vtkFloatArray()
velocity.SetNumberOfComponents(3)
velocity.SetName("Velocity")

for filename in glob.glob(fname):
    data = read_vtp(filename)
    pts_cnt = data.GetNumberOfPoints()

    if data is not None:

        print filename , pts_cnt
        for i in range(pts_cnt):
            pts.InsertNextPoint(data.GetPoint(i))

            velvalue = data.GetPointData().GetArray("Velocity").GetTuple3(i)
            velocity.InsertNextTuple3(velvalue[0],velvalue[1],velvalue[2])
        
pdata.SetPoints(pts)        
pdata.GetPointData().AddArray(velocity)
        
write_vtp(outfile,pdata)       
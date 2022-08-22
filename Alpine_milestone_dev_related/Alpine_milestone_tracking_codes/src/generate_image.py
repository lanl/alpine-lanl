## Post-hoc feature tracking algorithm development.
## Algorithm developed for ECP Alpine Project.
## This code is delivered as part of Alpine 16-14 P6 activity.
## Please contact Soumya Dutta (sdutta@lanl.gov) for any questions.
## Unauthorized use of this code is prohibited.

# trace generated using paraview version 5.8.0
#
#### import the simple module from the paraview

import numpy as np
import sys
import math
import os
import glob
from multiprocessing import Pool
from paraview.simple import *
##########################################

## this function helps to clean the memory usage for a pv script function. Not in use here
def ResetSession():
    pxm = servermanager.ProxyManager()
    pxm.UnRegisterProxies()
    del pxm
    Disconnect()
    Connect()

def generate_image_new(inparam):

	## parse params
	inputfile = inparam[2]
	tstep = inparam[1]
	idx = inparam[0]
	outpath = inparam[3]

	print ('processing ' + str(inparam))

	# create a new 'XML Image Data Reader'
	segmented_feature_20_178vti = XMLImageDataReader(FileName=[inputfile])
	segmented_feature_20_178vti.PointArrayStatus = ['feature_similarity']

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')
	# uncomment following to set a specific view size
	renderView1.ViewSize = [920, 1410]

	# get layout
	layout1 = GetLayout()

	# show data in view
	segmented_feature_20_178vtiDisplay = Show(segmented_feature_20_178vti, renderView1)

	
	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	segmented_feature_20_178vtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9938611944699142, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	segmented_feature_20_178vtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9938611944699142, 1.0, 0.5, 0.0]

	# init the 'Plane' selected for 'SliceFunction'
	#segmented_feature_20_178vtiDisplay.SliceFunction.Origin = [0.03938231895175164, 0.0014329018026482715, 0.025129255336778105]

	# reset view to fit data
	renderView1.ResetCamera()

	# get the material library
	materialLibrary1 = GetMaterialLibrary()

	# update the view to ensure updated data information
	renderView1.Update()

	# set scalar coloring
	ColorBy(segmented_feature_20_178vtiDisplay, ('POINTS', 'feature_similarity'))

	# rescale color and/or opacity maps used to include current data range
	segmented_feature_20_178vtiDisplay.RescaleTransferFunctionToDataRange(True, True)

	# change representation type
	segmented_feature_20_178vtiDisplay.SetRepresentationType('Volume')

	# get color transfer function/color map for 'feature_similarity'
	feature_similarityLUT = GetColorTransferFunction('feature_similarity')

	# get opacity transfer function/opacity map for 'feature_similarity'
	feature_similarityPWF = GetOpacityTransferFunction('feature_similarity')

	# create a new 'Resample To Image'
	resampleToImage1 = ResampleToImage(Input=segmented_feature_20_178vti)
	resampleToImage1.SamplingBounds = [0.0, 0.07876463790350328, 0.0, 0.002865803605296543, 0.0, 0.05025851067355621]

	# Properties modified on resampleToImage1
	resampleToImage1.SamplingDimensions = [512, 64, 512]

	# show data in view
	resampleToImage1Display = Show(resampleToImage1, renderView1)

	# Hide orientation axes
	renderView1.OrientationAxesVisibility = 0

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	resampleToImage1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	resampleToImage1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# hide data in view
	Hide(segmented_feature_20_178vti, renderView1)

	# show color bar/color legend
	resampleToImage1Display.SetScalarBarVisibility(renderView1, True)

	LoadPalette(paletteName='WhiteBackground')

	# update the view to ensure updated data information
	renderView1.Update()

	# change representation type
	resampleToImage1Display.SetRepresentationType('Volume')

	# set active source
	SetActiveSource(segmented_feature_20_178vti)

	# toggle 3D widget visibility (only when running from the GUI)
	Show3DWidgets(proxy=segmented_feature_20_178vtiDisplay)

	# toggle 3D widget visibility (only when running from the GUI)
	Hide3DWidgets(proxy=segmented_feature_20_178vtiDisplay)

	# show data in view
	segmented_feature_20_178vtiDisplay = Show(segmented_feature_20_178vti, renderView1)

	# show color bar/color legend
	segmented_feature_20_178vtiDisplay.SetScalarBarVisibility(renderView1, True)

	# change representation type
	segmented_feature_20_178vtiDisplay.SetRepresentationType('Point Gaussian')

	# change representation type
	segmented_feature_20_178vtiDisplay.SetRepresentationType('Outline')

	# hide color bar/color legend
	segmented_feature_20_178vtiDisplay.SetScalarBarVisibility(renderView1, False)

	# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
	feature_similarityLUT.ApplyPreset('Cold and Hot', True)

	# invert the transfer function
	feature_similarityLUT.InvertTransferFunction()

	# reset view to fit data
	renderView1.ResetCamera()

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5008332133293152, 0.05384615436196327, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5021340847015381, 0.05384615436196327, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5034348964691162, 0.05384615436196327, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5034348964691162, 0.04871794953942299, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5034348964691162, 0.043589744716882706, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5047357678413391, 0.043589744716882706, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5047357678413391, 0.03846153989434242, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5047357678413391, 0.03333333507180214, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5047357678413391, 0.03846153989434242, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5047357678413391, 0.043589744716882706, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5047357678413391, 0.03846153989434242, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.506036639213562, 0.03846153989434242, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]
	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.506036639213562, 0.03333333507180214, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	## this is very important to update the pipeline so bound can be queried
	#UpdatePipeline(proxy=resampleToImage1Display)
	bounds = resampleToImage1.GetDataInformation().GetBounds()
	#print (bounds)
	x_cam_val = 0.5*(bounds[1] - bounds[0])
	y_cam_val = 0.5*(bounds[3] - bounds[2])
	z_cam_val = 0.5*(bounds[5] - bounds[4])

	# current camera placement for renderView1
	renderView1.CameraPosition = [x_cam_val, 0.175, z_cam_val]
	renderView1.CameraFocalPoint = [x_cam_val, y_cam_val , z_cam_val]
	renderView1.CameraViewUp = [1, 0, 0]

	# ##current camera placement for renderView1
	# renderView1.CameraPosition = [0.04384809457289268, -0.1548475867339604, 0.0241919275719757]
	# renderView1.CameraFocalPoint = [0.03609634382581602, 0.11642615231833646, 0.025818953503435715]
	# renderView1.CameraViewUp = [0.9995919207487195, 0.02856048862475056, 0.0005389465662764407]
	# renderView1.CameraParallelScale = 0.07163203514932634

	#outpath = '../out/generated_images/'
	fname = outpath + 'bubble_' + str(idx) + '_' + str(tstep) + '.png'
	SaveScreenshot(fname, renderView1, ImageResolution=[460, 705])

	Delete(renderView1)

########################################################################################

inputpath = '../out/segmented_volumes/'
outpath = '../out/bubble_all.cdb/images/'

inputparams = []
for file in sorted(os.listdir(inputpath)):
    if file.endswith(".vti"):
    	fullname = os.path.join(inputpath,file)
    	fname = os.path.splitext(file)[0]
    	ffname = fname.split('_')
    	idx = ffname[2]
    	tstep = ffname[3]
    	print (idx,tstep,fullname,outpath)
    	inputparams = [idx,tstep,fullname,outpath]
    	generate_image_new(inputparams)

    	
print ('done processing all images')


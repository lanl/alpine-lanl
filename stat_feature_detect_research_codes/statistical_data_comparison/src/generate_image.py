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

## this function helps to clean the memory usage for a pv script function.
def ResetSession():
    pxm = servermanager.ProxyManager()
    pxm.UnRegisterProxies()
    del pxm
    Disconnect()
    Connect()

def generate_image_new(inparam):

	# trace generated using paraview version 5.8.0
	#
	# To ensure correct image size when batch processing, please search 
	# for and uncomment the line `# renderView*.ViewSize = [*,*]`


	#### import the simple module from the paraview
	#from paraview.simple import *
	#### disable automatic camera reset on 'Show'
	#paraview.simple._DisableFirstRenderCameraReset()

	## parse params
	inputfile = inparam[2]
	tstep = inparam[1]
	idx = inparam[0]

	print ('processing ' + str(inparam))

	# create a new 'XML Image Data Reader'
	segmented_feature_20_178vti = XMLImageDataReader(FileName=[inputfile])
	segmented_feature_20_178vti.PointArrayStatus = ['feature_similarity']

	# ## this is very important to update the pipeline so bound can be queried
	# UpdatePipeline(proxy=segmented_feature_20_178vti)
	# bounds = segmented_feature_20_178vti.GetDataInformation().GetBounds()
	# print (bounds)

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')
	# uncomment following to set a specific view size
	renderView1.ViewSize = [920, 1410]

	# get layout
	layout1 = GetLayout()

	# show data in view
	segmented_feature_20_178vtiDisplay = Show(segmented_feature_20_178vti, renderView1)

	# trace defaults for the display properties.
	segmented_feature_20_178vtiDisplay.Representation = 'Outline'
	segmented_feature_20_178vtiDisplay.ColorArrayName = [None, '']
	segmented_feature_20_178vtiDisplay.OSPRayScaleArray = 'feature_similarity'
	segmented_feature_20_178vtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
	segmented_feature_20_178vtiDisplay.SelectOrientationVectors = 'None'
	segmented_feature_20_178vtiDisplay.ScaleFactor = 0.007876463790350329
	segmented_feature_20_178vtiDisplay.SelectScaleArray = 'None'
	segmented_feature_20_178vtiDisplay.GlyphType = 'Arrow'
	segmented_feature_20_178vtiDisplay.GlyphTableIndexArray = 'None'
	segmented_feature_20_178vtiDisplay.GaussianRadius = 0.00039382318951751644
	segmented_feature_20_178vtiDisplay.SetScaleArray = ['POINTS', 'feature_similarity']
	segmented_feature_20_178vtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
	segmented_feature_20_178vtiDisplay.OpacityArray = ['POINTS', 'feature_similarity']
	segmented_feature_20_178vtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
	segmented_feature_20_178vtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
	segmented_feature_20_178vtiDisplay.PolarAxes = 'PolarAxesRepresentation'
	segmented_feature_20_178vtiDisplay.ScalarOpacityUnitDistance = 0.0015001675237156637
	#segmented_feature_20_178vtiDisplay.SliceFunction = 'Plane'
	segmented_feature_20_178vtiDisplay.Slice = 63

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

	# trace defaults for the display properties.
	resampleToImage1Display.Representation = 'Outline'
	resampleToImage1Display.ColorArrayName = ['POINTS', 'feature_similarity']
	resampleToImage1Display.LookupTable = feature_similarityLUT
	resampleToImage1Display.OSPRayScaleArray = 'feature_similarity'
	resampleToImage1Display.OSPRayScaleFunction = 'PiecewiseFunction'
	resampleToImage1Display.SelectOrientationVectors = 'None'
	resampleToImage1Display.ScaleFactor = 0.007876463790350329
	resampleToImage1Display.SelectScaleArray = 'None'
	resampleToImage1Display.GlyphType = 'Arrow'
	resampleToImage1Display.GlyphTableIndexArray = 'None'
	resampleToImage1Display.GaussianRadius = 0.00039382318951751644
	resampleToImage1Display.SetScaleArray = ['POINTS', 'feature_similarity']
	resampleToImage1Display.ScaleTransferFunction = 'PiecewiseFunction'
	resampleToImage1Display.OpacityArray = ['POINTS', 'feature_similarity']
	resampleToImage1Display.OpacityTransferFunction = 'PiecewiseFunction'
	resampleToImage1Display.DataAxesGrid = 'GridAxesRepresentation'
	resampleToImage1Display.PolarAxes = 'PolarAxesRepresentation'
	resampleToImage1Display.ScalarOpacityUnitDistance = 0.000367546148801177
	resampleToImage1Display.ScalarOpacityFunction = feature_similarityPWF
	#resampleToImage1Display.SliceFunction = 'Plane'
	resampleToImage1Display.Slice = 255

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	resampleToImage1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	resampleToImage1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# init the 'Plane' selected for 'SliceFunction'
	#resampleToImage1Display.SliceFunction.Origin = [0.03938231895175164, 0.0014329018026482715, 0.025129255336778105]

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
	#Show3DWidgets(proxy=segmented_feature_20_178vtiDisplay.SliceFunction)

	# toggle 3D widget visibility (only when running from the GUI)
	Show3DWidgets(proxy=segmented_feature_20_178vtiDisplay)

	# toggle 3D widget visibility (only when running from the GUI)
	#Hide3DWidgets(proxy=segmented_feature_20_178vtiDisplay.SliceFunction)

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

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7068024277687073, 0.684615433216095, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7068024277687073, 0.6794872283935547, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7119057178497314, 0.6487179398536682, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7871788740158081, 0.23846153914928436, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7884547114372253, 0.23333333432674408, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.791006326675415, 0.20769231021404266, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7922821640968323, 0.1974359005689621, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7922821640968323, 0.19230769574642181, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7922821640968323, 0.16153846681118011, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7922821640968323, 0.11538461595773697, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7922821640968323, 0.12051282078027725, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7922821640968323, 0.12564103305339813, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7922821640968323, 0.13076923787593842, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7922821640968323, 0.12564103305339813, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7922821640968323, 0.12051282078027725, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7935580015182495, 0.12051282078027725, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7935580015182495, 0.11538461595773697, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.794833779335022, 0.11538461595773697, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.794833779335022, 0.11025641113519669, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.794833779335022, 0.1051282063126564, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7961096167564392, 0.1051282063126564, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7986612319946289, 0.1051282063126564, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.8037645220756531, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.8063161373138428, 0.10000000149011612, 0.5, 0.0, 0.9938611944699144, 1.0, 0.5, 0.0]

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

	print (x_cam_val,y_cam_val,z_cam_val)

	# current camera placement for renderView1
	renderView1.CameraPosition = [x_cam_val, 0.175, z_cam_val]
	renderView1.CameraFocalPoint = [x_cam_val, y_cam_val , z_cam_val]
	renderView1.CameraViewUp = [1, 0, 0]
	#renderView1.CameraParallelScale = 0.07163203514932634

	# ##current camera placement for renderView1
	# renderView1.CameraPosition = [0.04384809457289268, -0.1548475867339604, 0.0241919275719757]
	# renderView1.CameraFocalPoint = [0.03609634382581602, 0.11642615231833646, 0.025818953503435715]
	# renderView1.CameraViewUp = [0.9995919207487195, 0.02856048862475056, 0.0005389465662764407]
	# renderView1.CameraParallelScale = 0.07163203514932634

	# save screenshot
	#SaveScreenshot('/Users/sdutta/Desktop/a.png', renderView1, ImageResolution=[920, 1410], CompressionLevel='2')

	#outpath = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_feature_images/'
	outpath = '/Users/sdutta/Desktop/test/'
	fname = outpath + 'bubble_' + str(idx) + '_' + str(tstep) + '.png'
	#SaveScreenshot(fname, magnification=1, quality=100, view=renderView1)
	SaveScreenshot(fname, renderView1, ImageResolution=[920, 1410], CompressionLevel='2')

	Delete(renderView1)

def generate_image(inparam):

	## parse params
	inputfile = inparam[2]
	tstep = inparam[1]
	idx = inparam[0]

	print ('processing ' + str(inparam))

	#ResetSession()

	#### disable automatic camera reset on 'Show'
	paraview.simple._DisableFirstRenderCameraReset()

	LoadPalette(paletteName='WhiteBackground')

	# create a new 'XML Image Data Reader'
	segmented_feature_1_75vti = XMLImageDataReader(FileName=[inputfile])
	segmented_feature_1_75vti.PointArrayStatus = ['feature_similarity']

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')
	# uncomment following to set a specific view size
	renderView1.ViewSize = [1292, 1410]

	# get layout
	layout1 = GetLayout()

	# show data in view
	segmented_feature_1_75vtiDisplay = Show(segmented_feature_1_75vti, renderView1)

	# trace defaults for the display properties.
	segmented_feature_1_75vtiDisplay.Representation = 'Outline'
	segmented_feature_1_75vtiDisplay.ColorArrayName = [None, '']
	segmented_feature_1_75vtiDisplay.OSPRayScaleArray = 'feature_similarity'
	segmented_feature_1_75vtiDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
	segmented_feature_1_75vtiDisplay.SelectOrientationVectors = 'None'
	segmented_feature_1_75vtiDisplay.ScaleFactor = 0.0074684731426524365
	segmented_feature_1_75vtiDisplay.SelectScaleArray = 'None'
	segmented_feature_1_75vtiDisplay.GlyphType = 'Arrow'
	segmented_feature_1_75vtiDisplay.GlyphTableIndexArray = 'None'
	segmented_feature_1_75vtiDisplay.GaussianRadius = 0.0003734236571326218
	segmented_feature_1_75vtiDisplay.SetScaleArray = ['POINTS', 'feature_similarity']
	segmented_feature_1_75vtiDisplay.ScaleTransferFunction = 'PiecewiseFunction'
	segmented_feature_1_75vtiDisplay.OpacityArray = ['POINTS', 'feature_similarity']
	segmented_feature_1_75vtiDisplay.OpacityTransferFunction = 'PiecewiseFunction'
	segmented_feature_1_75vtiDisplay.DataAxesGrid = 'GridAxesRepresentation'
	segmented_feature_1_75vtiDisplay.PolarAxes = 'PolarAxesRepresentation'
	segmented_feature_1_75vtiDisplay.ScalarOpacityUnitDistance = 0.0014454263796353482
	#segmented_feature_1_75vtiDisplay.SliceFunction = 'Plane'
	segmented_feature_1_75vtiDisplay.Slice = 63

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	segmented_feature_1_75vtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9998450847203422, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	segmented_feature_1_75vtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9998450847203422, 1.0, 0.5, 0.0]

	# init the 'Plane' selected for 'SliceFunction'
	#segmented_feature_1_75vtiDisplay.SliceFunction.Origin = [0.03734236571326218, 0.001433464239789397, 0.025129195339198215]

	# reset view to fit data
	renderView1.ResetCamera()

	# get the material library
	materialLibrary1 = GetMaterialLibrary()

	# update the view to ensure updated data information
	renderView1.Update()

	# set scalar coloring
	ColorBy(segmented_feature_1_75vtiDisplay, ('POINTS', 'feature_similarity'))

	# rescale color and/or opacity maps used to include current data range
	segmented_feature_1_75vtiDisplay.RescaleTransferFunctionToDataRange(True, True)

	# change representation type
	segmented_feature_1_75vtiDisplay.SetRepresentationType('Volume')

	# get color transfer function/color map for 'feature_similarity'
	feature_similarityLUT = GetColorTransferFunction('feature_similarity')

	# get opacity transfer function/opacity map for 'feature_similarity'
	feature_similarityPWF = GetOpacityTransferFunction('feature_similarity')

	# reset view to fit data
	renderView1.ResetCamera()

	# reset view to fit data bounds
	renderView1.ResetCamera(0.0, 0.0746847314265, 0.0, 0.00286692847958, 0.0, 0.0502583906784)

	# Hide orientation axes
	renderView1.OrientationAxesVisibility = 0

	# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
	feature_similarityLUT.ApplyPreset('Cold and Hot', True)

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.9998450847203422, 0.992307722568512, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6420058012008667, 0.6025640964508057, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6420058012008667, 0.5923076868057251, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6399008631706238, 0.5512820482254028, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6399008631706238, 0.535897433757782, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6399008631706238, 0.5205128192901611, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6399008631706238, 0.5102564096450806, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6399008631706238, 0.5, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6399008631706238, 0.47948718070983887, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6399008631706238, 0.464102566242218, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6399008631706238, 0.45384615659713745, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.4435897469520569, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.4333333373069763, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.42307692766189575, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.4076923131942749, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.4025641083717346, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.39230769872665405, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.3717948794364929, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.36666667461395264, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.33589744567871094, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.32564103603363037, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.3205128312110901, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6377959251403809, 0.3102564215660095, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6356909871101379, 0.30512821674346924, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6272712349891663, 0.29487180709838867, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.6167465448379517, 0.2897436022758484, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.60832679271698, 0.2897436022758484, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5935922861099243, 0.2897436022758484, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5914873480796814, 0.2846153974533081, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5283392071723938, 0.2589743733406067, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5030799508094788, 0.2538461685180664, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5009750127792358, 0.2538461685180664, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.454666405916214, 0.21794871985912323, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45256146788597107, 0.21794871985912323, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.21282051503658295, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.20769231021404266, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.20256410539150238, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.1974359005689621, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.19230769574642181, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18205128610134125, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.9871795177459717, 0.5, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.4999225423601711, 0.0, 0.0, 0.501960784314, 0.549388587474823, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.4999225423601711, 0.0, 0.0, 0.501960784314, 0.5535984635353088, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.4999225423601711, 0.0, 0.0, 0.501960784314, 0.5683330297470093, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.4999225423601711, 0.0, 0.0, 0.501960784314, 0.576752781867981, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.4999225423601711, 0.0, 0.0, 0.501960784314, 0.5978021621704102, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.4999225423601711, 0.0, 0.0, 0.501960784314, 0.6188514828681946, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.4999225423601711, 0.0, 0.0, 0.501960784314, 0.6209564208984375, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.4999225423601711, 0.0, 0.0, 0.501960784314, 0.6377959251403809, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.4999225423601711, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5093947649002075, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5114997029304504, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5157095789909363, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5199194550514221, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.522024393081665, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.524129331111908, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5262342691421509, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5451787114143372, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.549388587474823, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5557034015655518, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5620182156562805, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.574647843837738, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5809626579284668, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.660950243473053, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.6672650575637817, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.6735798716545105, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.6756848096847534, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.6925243139266968, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.6946292519569397, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.6967341899871826, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.7030490040779114, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.7051539421081543, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.7072588801383972, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.7051539421081543, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.7030490040779114, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.7009440660476685, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.6988391280174255, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.6967341899871826, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.6946292519569397, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 0.0, 1.0, 1.0, 0.449930288124154, 0.0, 0.0, 1.0, 0.5830675959587097, 0.0, 0.0, 0.501960784314, 0.6925243139266968, 1.0, 0.0, 0.0, 0.9998450847203422, 1.0, 1.0, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.9717949032783508, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.9666666984558105, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.95641028881073, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.9461538791656494, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.9153846502304077, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.9051282405853271, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.8897436261177063, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.8794872164726257, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.8743590116500854, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.8794872164726257, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.884615421295166, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.8948718309402466, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.9000000357627869, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.45045652985572815, 0.18717949092388153, 0.5, 0.0, 0.9998450847203422, 0.9051282405853271, 0.5, 0.0]

	#### saving camera placements for all active views

	# current camera placement for renderView1
	renderView1.CameraPosition = [0.037245832926616575, 0.1506845345035667, 0.02438948603570985]
	renderView1.CameraFocalPoint = [0.03734678465639598, -0.005398742879353881, 0.025163056718955558]
	renderView1.CameraViewUp = [0.9999991478222848, 0.0006523851418458994, 0.0011308175498130901]
	renderView1.CameraParallelScale = 0.049146072105381564

	#### uncomment the following to render all views
	# RenderAllViews()
	outpath = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_feature_images/'
	fname = outpath + 'bubble_' + str(idx) + '_' + str(tstep) + '.png'
	SaveScreenshot(fname, magnification=1, quality=100, view=renderView1)

	Delete(renderView1)

########################################################################################

inputpath = '/Users/sdutta/Desktop/test_image_generation/'
# initstep=75
# finalstep=76

inputparams = []
for file in sorted(os.listdir(inputpath)):
    if file.endswith(".vti"):
    	fullname = os.path.join(inputpath,file)
    	fname = os.path.splitext(file)[0]
    	ffname = fname.split('_')
    	idx = ffname[2]
    	tstep = ffname[3]
    	print (idx,tstep,fullname)
    	inputparams.append([idx,tstep,fullname])
    	#generate_image(idx,tstep,fullname)


# Create a pool of worker processes, each able to use a CPU core
pool = Pool(processes=2)
args = [(inputparams[i]) for i in range(0,len(inputparams))]

#print (args)

# ## Execute the multiprocess code
pool.map(generate_image_new, args)

print ('done processing all images')


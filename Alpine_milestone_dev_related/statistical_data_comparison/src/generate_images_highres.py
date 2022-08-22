# trace generated using paraview version 5.8.0
import numpy as np
import sys
import math
import os
import glob
from multiprocessing import Pool
from paraview.simple import *

## this function helps to clean the memory usage for a pv script function.
def ResetSession():
    pxm = servermanager.ProxyManager()
    pxm.UnRegisterProxies()
    del pxm
    Disconnect()
    Connect()

def generate_image(idx,tstep,fullname):

	#### disable automatic camera reset on 'Show'
	#paraview.simple._DisableFirstRenderCameraReset()

	ResetSession()

	# create a new 'XML Image Data Reader'
	segmented_feature_8_12030vti = XMLImageDataReader(FileName=[fullname])
	segmented_feature_8_12030vti.PointArrayStatus = ['feature_similarity']

	# get active view
	renderView1 = GetActiveViewOrCreate('RenderView')
	# uncomment following to set a specific view size
	renderView1.ViewSize = [570, 701]

	# get layout
	layout1 = GetLayout()

	# show data in view
	segmented_feature_8_12030vtiDisplay = Show(segmented_feature_8_12030vti, renderView1, 'UniformGridRepresentation')

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	segmented_feature_8_12030vtiDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.999937117099762, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	segmented_feature_8_12030vtiDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.999937117099762, 1.0, 0.5, 0.0]

	# init the 'Plane' selected for 'SliceFunction'
	segmented_feature_8_12030vtiDisplay.SliceFunction.Origin = [0.046897389612174996, 0.016395527345575, 0.056388697999144005]

	# reset view to fit data
	renderView1.ResetCamera()

	# get the material library
	materialLibrary1 = GetMaterialLibrary()

	# update the view to ensure updated data information
	renderView1.Update()

	# set scalar coloring
	ColorBy(segmented_feature_8_12030vtiDisplay, ('POINTS', 'feature_similarity'))

	# rescale color and/or opacity maps used to include current data range
	segmented_feature_8_12030vtiDisplay.RescaleTransferFunctionToDataRange(True, True)

	# change representation type
	segmented_feature_8_12030vtiDisplay.SetRepresentationType('Volume')

	# get color transfer function/color map for 'feature_similarity'
	feature_similarityLUT = GetColorTransferFunction('feature_similarity')

	# get opacity transfer function/opacity map for 'feature_similarity'
	feature_similarityPWF = GetOpacityTransferFunction('feature_similarity')

	# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
	feature_similarityLUT.ApplyPreset('Cold and Hot', True)

	# invert the transfer function
	feature_similarityLUT.InvertTransferFunction()

	# create a new 'Resample To Image'
	resampleToImage1 = ResampleToImage(Input=segmented_feature_8_12030vti)
	resampleToImage1.SamplingBounds = [7.312723028e-05, 0.09372165199407, 0.0053990324959, 0.02739202219525, 7.2988899774e-05, 0.11270440709851401]

	# Properties modified on resampleToImage1
	resampleToImage1.SamplingDimensions = [512, 64, 512]

	# show data in view
	resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	resampleToImage1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	resampleToImage1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# init the 'Plane' selected for 'SliceFunction'
	resampleToImage1Display.SliceFunction.Origin = [0.046897389612174996, 0.016395527345575, 0.056388697999144005]

	# hide data in view
	Hide(segmented_feature_8_12030vti, renderView1)

	# show color bar/color legend
	resampleToImage1Display.SetScalarBarVisibility(renderView1, True)

	# update the view to ensure updated data information
	renderView1.Update()

	# hide color bar/color legend
	resampleToImage1Display.SetScalarBarVisibility(renderView1, False)

	# change representation type
	resampleToImage1Display.SetRepresentationType('Volume')

	# Hide orientation axes
	renderView1.OrientationAxesVisibility = 0

	# reset view to fit data
	renderView1.ResetCamera()

	# reset view to fit data
	renderView1.ResetCamera()

	# reset view to fit data bounds
	renderView1.ResetCamera(7.312723028e-05, 0.0937216519941, 0.0053990324959, 0.0273920221952, 7.2988899774e-05, 0.112704407099)

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.5499654144048692, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.4280667304992676, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.5499654144048692, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.23409900069236755, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.5499654144048692, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.29095160961151123, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.5499654144048692, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.5499654144048692, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.5618375539779663, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.5651818513870239, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.5852474570274353, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.6019688248634338, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.6053131222724915, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.6086573600769043, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.6120016574859619, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.6153459548950195, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.62203449010849, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.6287230253219604, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.6387557983398438, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49996855854988104, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.49495214223861694, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.504984974861145, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5116735100746155, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.528394877910614, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5350834131240845, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5451162457466125, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5484604835510254, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18393492698669434, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.1805906593799591, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18393492698669434, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2040005475282669, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2976401448249817, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.3009844124317169, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.30767297744750977, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.311017245054245, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.3143615126609802, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.311017245054245, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.30767297744750977, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.30432868003845215, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2976401448249817, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.29429587721824646, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.29095160961151123, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.287607342004776, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.28426307439804077, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.28091877698898315, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2775745093822479, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2742302417755127, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.27088597416877747, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.26754170656204224, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.264197438955307, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.25750890374183655, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2541646361351013, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2508203387260437, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.24747607111930847, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.24413180351257324, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.240787535905838, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.23744326829910278, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.23075471818447113, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2274104505777359, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.22072190046310425, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5250505805015564, 0.04278074949979782, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5250505805015564, 0.03743315488100052, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5250505805015564, 0.026737969368696213, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5250505805015564, 0.02139037474989891, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.22406618297100067, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2274104505777359, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.23075471818447113, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.23409900069236755, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.23744326829910278, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.240787535905838, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.24413180351257324, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.24747607111930847, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2508203387260437, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2541646361351013, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.25750890374183655, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2608531713485718, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.264197438955307, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2608531713485718, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.25750890374183655, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.2541646361351013, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.240787535905838, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.23744326829910278, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.23409900069236755, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.19731201231479645, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.19396772980690002, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.1906234622001648, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.8728548288345337, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.8661662936210632, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.8494449257850647, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.8461006283760071, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.8394120931625366, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.8360678553581238, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.8293793201446533, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.7925922870635986, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.7892480492591858, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.7859037518501282, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.7825595140457153, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityLUT
	feature_similarityLUT.RGBPoints = [0.0, 1.0, 1.0, 0.0, 0.18727919459342957, 1.0, 0.5882352941176471, 0.0, 0.44813236594200134, 1.0, 0.0, 0.0, 0.5885917544364929, 0.0, 0.0, 0.501960784314, 0.6421000957489014, 0.0, 0.0, 1.0, 0.7792152166366577, 0.0, 0.6509803921568628, 1.0, 0.9999371170997621, 0.0, 1.0, 1.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8761990666389465, 0.7005347609519958, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8728548288345337, 0.6631016135215759, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8661662936210632, 0.4545454680919647, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8661662936210632, 0.4117647111415863, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8594777584075928, 0.3743315637111664, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8561334609985352, 0.32085561752319336, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8561334609985352, 0.31016042828559875, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8461006283760071, 0.33689841628074646, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8394120931625366, 0.3796791434288025, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8360678553581238, 0.385026752948761, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8293793201446533, 0.3903743326663971, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8327235579490662, 0.3903743326663971, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8795433640480042, 0.3957219421863556, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8862318992614746, 0.385026752948761, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8862318992614746, 0.34224599599838257, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8862318992614746, 0.33689841628074646, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8862318992614746, 0.32620322704315186, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8795433640480042, 0.29411765933036804, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8795433640480042, 0.29946523904800415, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8761990666389465, 0.30481284856796265, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8761990666389465, 0.31016042828559875, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8695105314254761, 0.32620322704315186, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8594777584075928, 0.3529411852359772, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# Properties modified on feature_similarityPWF
	feature_similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.1571807563304901, 0.0, 0.5, 0.0, 0.5250505805015564, 0.01604278013110161, 0.5, 0.0, 0.6186901926994325, 0.06951871514320374, 0.5, 0.0, 0.8561334609985352, 0.3636363744735718, 0.5, 0.0, 0.9999371170997621, 1.0, 0.5, 0.0]

	# create a new 'XML Image Data Reader'
	segmented_feature_8_12030vti_1 = XMLImageDataReader(FileName=[fullname])
	segmented_feature_8_12030vti_1.PointArrayStatus = ['feature_similarity']

	# show data in view
	segmented_feature_8_12030vti_1Display = Show(segmented_feature_8_12030vti_1, renderView1, 'UniformGridRepresentation')

	# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
	segmented_feature_8_12030vti_1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.999937117099762, 1.0, 0.5, 0.0]

	# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
	segmented_feature_8_12030vti_1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.999937117099762, 1.0, 0.5, 0.0]

	# init the 'Plane' selected for 'SliceFunction'
	segmented_feature_8_12030vti_1Display.SliceFunction.Origin = [0.046897389612174996, 0.016395527345575, 0.056388697999144005]

	LoadPalette(paletteName='WhiteBackground')

	# update the view to ensure updated data information
	renderView1.Update()

	# Properties modified on segmented_feature_8_12030vti_1Display
	segmented_feature_8_12030vti_1Display.LineWidth = 2.0

	# reset view to fit data
	renderView1.ResetCamera()

	# reset view to fit data bounds
	renderView1.ResetCamera(7.312723028e-05, 0.0937216519941, 0.0053990324959, 0.0273920221952, 7.2988899774e-05, 0.112704407099)

	# current camera placement for renderView1
	renderView1.CameraPosition = [0.038650960686393024, -0.1944837755872127, 0.05209757255680584]
	renderView1.CameraFocalPoint = [0.04982983999478375, 0.09138472547424996, 0.057914632747197714]
	renderView1.CameraViewUp = [0.9992351932702201, -0.039024846963911446, -0.002467762077082742]
	renderView1.CameraParallelScale = 0.07406006709389576

	# save screenshot

	outpath = '/Users/sdutta/Codes/statistical_data_comparison/out/segmented_images_highres/'
	fname = outpath + 'bubble_' + str(idx) + '_' + str(tstep) + '.png'

	SaveScreenshot(fname, renderView1, ImageResolution=[630, 522], CompressionLevel='2')


########################################################################################

inputpath = '/Users/sdutta/Codes/statistical_data_comparison/out/segmented_volumes_highres/'

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
    	generate_image(idx,tstep,fullname)




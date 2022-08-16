#--------------------------------------------------------------

# Global timestep output options
timeStepToStartOutputAt=0
forceOutputAtFirstCall=False

# Global screenshot output options
imageFileNamePadding=0
rescale_lookuptable=False

# Whether or not to request specific arrays from the adaptor.
requestSpecificArrays=False

# a root directory under which all Catalyst output goes
rootDirectory=''

# makes a cinema D index table
make_cinema_table=False

#--------------------------------------------------------------
# Code generated from cpstate.py to create the CoProcessor.
# paraview version 5.8.1-1-g40b41376db
#--------------------------------------------------------------

from paraview.simple import *
from paraview import coprocessing

# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      # state file generated using paraview version 5.8.1-1-g40b41376db

      # ----------------------------------------------------------------
      # setup views used in the visualization
      # ----------------------------------------------------------------

      # trace generated using paraview version 5.8.1-1-g40b41376db
      #
      # To ensure correct image size when batch processing, please search 
      # for and uncomment the line `# renderView*.ViewSize = [*,*]`

      #### disable automatic camera reset on 'Show'
      paraview.simple._DisableFirstRenderCameraReset()

      # get the material library
      materialLibrary1 = GetMaterialLibrary()

      # Create a new 'Render View'
      renderView1 = CreateView('RenderView')
      renderView1.ViewSize = [716, 352]
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.CenterOfRotation = [0.0020009127283896633, 0.001999172356439815, 0.0019999934867559444]
      renderView1.StereoType = 'Crystal Eyes'
      renderView1.CameraPosition = [0.036644586455975216, 0.21705065997450937, 0.0662849960547136]
      renderView1.CameraFocalPoint = [0.054171492461448235, -0.2147265944903157, 0.06031747111213411]
      renderView1.CameraViewUp = [0.9991579942858331, 0.04046464155513918, 0.0067760784031188894]
      renderView1.CameraParallelScale = 0.11422577498907434      
      renderView1.BackEnd = 'OSPRay raycaster'
      renderView1.OSPRayMaterialLibrary = materialLibrary1

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='RenderView1_%t.png', freq=1, fittoscreen=0, magnification=1, width=716, height=352, cinema={}, compression=5)
      renderView1.ViewTime = datadescription.GetTime()

      SetActiveView(None)

      # ----------------------------------------------------------------
      # setup view layouts
      # ----------------------------------------------------------------

      # create new layout object 'Layout #1'
      layout1 = CreateLayout(name='Layout #1')
      layout1.AssignView(0, renderView1)

      # ----------------------------------------------------------------
      # restore active view
      SetActiveView(renderView1)
      # ----------------------------------------------------------------

      # ----------------------------------------------------------------
      # setup the data processing pipelines
      # ----------------------------------------------------------------

      # create a new 'AMReX/BoxLib Particles Reader'
      # create a producer from a simulation input
      dEM07_plt00050 = coprocessor.CreateProducer(datadescription, 'inputparticles')

      # create a new 'Feature Analysis'
      featureAnalysis1 = FeatureAnalysis(Input=dEM07_plt00050)
      featureAnalysis1.ClusterBlockSize = [3, 3, 3]
      featureAnalysis1.FeatureGaussian = [2.0, 10.0]

      # create a new 'Resample To Image'
      resampleToImage1 = ResampleToImage(Input=featureAnalysis1)
      resampleToImage1.SamplingBounds = [1.9417382859558333e-06, 0.003984164024624505, 1.748457857490837e-08, 0.003865203601139793, 2.523888219899246e-06, 0.003966030690060129]

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from resampleToImage1
      resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')

      # get color transfer function/color map for 'similarity'
      similarityLUT = GetColorTransferFunction('similarity')
      similarityLUT.ApplyPreset('Cold and Hot', True) ####
      similarityLUT.InvertTransferFunction() ####
      similarityLUT.ScalarRangeInitialized = 1.0

      # get opacity transfer function/opacity map for 'similarity'
      similarityPWF = GetOpacityTransferFunction('similarity')i
      similarityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.37078651785850525, 0.02139037474989891, 0.5, 0.0, 0.7387640476226807, 0.04812834411859512, 0.5, 0.0, 0.8904494643211365, 0.27272728085517883, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
      ##similarityPWF.Points = [0.0, 1.0, 0.5, 0.0, 0.25070422887802124, 0.3375000059604645, 0.5, 0.0, 0.490140825510025, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
      similarityPWF.ScalarRangeInitialized = 1

      # trace defaults for the display properties.
      resampleToImage1Display.Representation = 'Volume'
      resampleToImage1Display.ColorArrayName = ['POINTS', 'similarity']
      resampleToImage1Display.LookupTable = similarityLUT
      resampleToImage1Display.OSPRayScaleArray = 'similarity'
      resampleToImage1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      resampleToImage1Display.SelectOrientationVectors = 'None'
      resampleToImage1Display.ScaleFactor = 0.0003982222286338549
      resampleToImage1Display.SelectScaleArray = 'None'
      resampleToImage1Display.GlyphType = 'Arrow'
      resampleToImage1Display.GlyphTableIndexArray = 'None'
      resampleToImage1Display.GaussianRadius = 1.9911111431692744e-05
      resampleToImage1Display.SetScaleArray = ['POINTS', 'similarity']
      resampleToImage1Display.ScaleTransferFunction = 'PiecewiseFunction'
      resampleToImage1Display.OpacityArray = ['POINTS', 'similarity']
      resampleToImage1Display.OpacityTransferFunction = 'PiecewiseFunction'
      resampleToImage1Display.DataAxesGrid = 'GridAxesRepresentation'
      resampleToImage1Display.PolarAxes = 'PolarAxesRepresentation'
      resampleToImage1Display.ScalarOpacityUnitDistance = 6.888499664772514e-05
      resampleToImage1Display.ScalarOpacityFunction = similarityPWF
      resampleToImage1Display.SliceFunction = 'Plane'
      resampleToImage1Display.Slice = 49

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      resampleToImage1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9267358779907227, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      resampleToImage1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.9267358779907227, 1.0, 0.5, 0.0]

      # init the 'Plane' selected for 'SliceFunction'
      resampleToImage1Display.SliceFunction.Origin = [0.0019930528814552304, 0.0019326105428591842, 0.0019842772891400145]

      # setup the color legend parameters for each legend in this view

      # get color legend/bar for similarityLUT in view renderView1
      similarityLUTColorBar = GetScalarBar(similarityLUT, renderView1)
      similarityLUTColorBar.Title = 'similarity'
      similarityLUTColorBar.ComponentTitle = ''
      similarityLUTColorBar.Orientation = 'Vertical' ##
      similarityLUTColorBar.Position = [0.862998295615945, 0.13773147335584612] ##
      similarityLUTColorBar.ScalarBarLength = 0.33000000000000007 ##

      # set color bar visibility
      similarityLUTColorBar.Visibility = 1

      # show color legend
      resampleToImage1Display.SetScalarBarVisibility(renderView1, True)

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(resampleToImage1)
      # ----------------------------------------------------------------

      # Now any catalyst writers
      xMLPImageDataWriter1 = servermanager.writers.XMLPImageDataWriter(Input=resampleToImage1)
      coprocessor.RegisterWriter(xMLPImageDataWriter1, filename='ResampleToImage1_%t.pvti', freq=1, paddingamount=0, DataMode='Appended', HeaderType='UInt64', EncodeAppendedData=False, CompressorType='None', CompressionLevel='6')

    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'inputparticles': [1]}
  coprocessor.SetUpdateFrequencies(freqs)
  if requestSpecificArrays:
    arrays = [['cpu', 0], ['density', 0], ['dragx', 0], ['dragy', 0], ['dragz', 0], ['id', 0], ['mass', 0], ['omegax', 0], ['omegay', 0], ['omegaz', 0], ['omoi', 0], ['phase', 0], ['radius', 0], ['state', 0], ['velx', 0], ['vely', 0], ['velz', 0], ['volume', 0]]
    coprocessor.SetRequestedArrays('inputparticles', arrays)
  coprocessor.SetInitialOutputOptions(timeStepToStartOutputAt,forceOutputAtFirstCall)

  if rootDirectory:
      coprocessor.SetRootDirectory(rootDirectory)

  if make_cinema_table:
      coprocessor.EnableCinemaDTable()

  return coprocessor


#--------------------------------------------------------------
# Global variable that will hold the pipeline for each timestep
# Creating the CoProcessor object, doesn't actually create the ParaView pipeline.
# It will be automatically setup when coprocessor.UpdateProducers() is called the
# first time.
coprocessor = CreateCoProcessor()

#--------------------------------------------------------------
# Enable Live-Visualizaton with ParaView and the update frequency
coprocessor.EnableLiveVisualization(False, 1)

# ---------------------- Data Selection method ----------------------

def RequestDataDescription(datadescription):
    "Callback to populate the request for current timestep"
    global coprocessor

    # setup requests for all inputs based on the requirements of the
    # pipeline.
    coprocessor.LoadRequestedData(datadescription)

# ------------------------ Processing method ------------------------

def DoCoProcessing(datadescription):
    "Callback to do co-processing for current timestep"
    global coprocessor

    # Update the coprocessor by providing it the newly generated simulation data.
    # If the pipeline hasn't been setup yet, this will setup the pipeline.
    coprocessor.UpdateProducers(datadescription)

    # Write output data, if appropriate.
    coprocessor.WriteData(datadescription);

    # Write image capture (Last arg: rescale lookup table), if appropriate.
    coprocessor.WriteImages(datadescription, rescale_lookuptable=rescale_lookuptable,
        image_quality=0, padding_amount=imageFileNamePadding)

    # Live Visualization, if enabled.
    coprocessor.DoLiveVisualization(datadescription, "localhost", 22222)
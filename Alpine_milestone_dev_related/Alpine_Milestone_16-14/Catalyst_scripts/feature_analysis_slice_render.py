
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
      renderView1.ViewSize = [534, 674]
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.CenterOfRotation = [0.0020009127283896633, 0.001999172356439815, 0.0019999934867559444]
      renderView1.StereoType = 'Crystal Eyes'
      renderView1.CameraPosition = [0.031401023308701585, 0.25028559636593944, -0.014164479279071853]
      renderView1.CameraFocalPoint = [0.03654233645213228, 0.06954866779289456, 0.0173019782925154]
      renderView1.CameraViewUp = [0.9994798180976283, 0.030336906395180968, 0.010943734550051676]
      renderView1.CameraParallelScale = 0.047500458236496935
      renderView1.BackEnd = 'OSPRay raycaster'
      renderView1.OSPRayMaterialLibrary = materialLibrary1


      LoadPalette(paletteName='WhiteBackground')

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='SimFieldSlice_%t.png', freq=1, fittoscreen=0, magnification=1, width=534, height=674, cinema={}, compression=5)
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

      ## ----------------------------------------------------------------
      ## setup the visualization in view 'renderView1'
      ## ----------------------------------------------------------------
      ## Now any catalyst writers
      ## Store the pvti version of similarity field out. Uncomment following two lines to store data.
      #xMLPImageDataWriter1 = servermanager.writers.XMLPImageDataWriter(Input=resampleToImage1)
      #coprocessor.RegisterWriter(xMLPImageDataWriter1, filename='ResampleToImage1_%t.pvti', freq=1, paddingamount=0, DataMode='Appended', HeaderType='UInt64', EncodeAppendedData=False, CompressorType='None', CompressionLevel='6')

      #################################################
      ## Do slice rendering
      ################################################
      # create a new 'Slice'
      slice1 = Slice(Input=resampleToImage1)
      slice1.SliceType = 'Plane'
      slice1.HyperTreeGridSlicer = 'Plane'
      slice1.SliceOffsetValues = [0.0]
      # init the 'Plane' selected for 'SliceType'
      slice1.SliceType.Origin = [0.03832189351188268, 0.006990435533225536, 0.028193420851529816]
      # init the 'Plane' selected for 'HyperTreeGridSlicer'
      slice1.HyperTreeGridSlicer.Origin = [0.03832189351188268, 0.006990435533225536, 0.028193420851529816]
      # Properties modified on slice1.SliceType
      slice1.SliceType.Normal = [0.0, 1.0, 0.0]
      # show data in view
      slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')
      # trace defaults for the display properties.
      slice1Display.Representation = 'Surface'
      slice1Display.ColorArrayName = [None, '']
      slice1Display.OSPRayScaleArray = 'similarity'
      slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      slice1Display.SelectOrientationVectors = 'None'
      slice1Display.ScaleFactor = 0.007649656653666171
      slice1Display.SelectScaleArray = 'None'
      slice1Display.GlyphType = 'Arrow'
      slice1Display.GlyphTableIndexArray = 'None'
      slice1Display.GaussianRadius = 0.00038248283268330855
      slice1Display.SetScaleArray = ['POINTS', 'similarity']
      slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
      slice1Display.OpacityArray = ['POINTS', 'similarity']
      slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
      slice1Display.DataAxesGrid = 'GridAxesRepresentation'
      slice1Display.PolarAxes = 'PolarAxesRepresentation'
      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      slice1Display.ScaleTransferFunction.Points = [0.25807446241378784, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      slice1Display.OpacityTransferFunction.Points = [0.25807446241378784, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
      # update the view to ensure updated data information
      renderView1.Update()
      # set scalar coloring
      ColorBy(slice1Display, ('POINTS', 'similarity'))
      # rescale color and/or opacity maps used to include current data range
      slice1Display.RescaleTransferFunctionToDataRange(True, False)
      ## show color bar/color legend
      #slice1Display.SetScalarBarVisibility(renderView1, True)
      # get color transfer function/color map for 'similarity'
      similarityLUT = GetColorTransferFunction('similarity')
      # get opacity transfer function/opacity map for 'similarity'
      similarityPWF = GetOpacityTransferFunction('similarity')

      SetActiveSource(resampleToImage1) ## is this needed?


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

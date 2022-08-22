
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
# paraview version 5.8.0
#--------------------------------------------------------------

from paraview.simple import *
from paraview import coprocessing

# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      # state file generated using paraview version 5.8.0

      # ----------------------------------------------------------------
      # setup views used in the visualization
      # ----------------------------------------------------------------

      # trace generated using paraview version 5.8.0
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

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='SimFieldParticle_%t.png', freq=1, fittoscreen=0, magnification=1, width=534, height=674, cinema={}, compression=5)
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
      rlt0000 = coprocessor.CreateProducer(datadescription, 'inputparticles')

      # create a new 'Calculator'
      calculator1 = Calculator(Input=rlt0000)
      calculator1.Function = 'sqrt(velx*velx+vely*vely+velz*velz)'
      
      # show data from calculator1
      calculator1Display = Show(calculator1, renderView1, 'GeometryRepresentation')

      # get color transfer function/color map for 'Result'
      resultLUT = GetColorTransferFunction('Result')
      resultLUT.AutomaticRescaleRangeMode = 'Never'
      resultLUT.RGBPoints = [0.00013329293792503272, 0.231373, 0.298039, 0.752941, 0.25006664646896254, 0.865003, 0.865003, 0.865003, 0.5, 0.705882, 0.0156863, 0.14902]
      resultLUT.ScalarRangeInitialized = 1.0

      # trace defaults for the display properties.
      calculator1Display.Representation = 'Point Gaussian' #'3D Glyphs'
      calculator1Display.ColorArrayName = ['POINTS', 'Result']
      calculator1Display.LookupTable = resultLUT
      calculator1Display.OSPRayScaleArray = 'Result'
      calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      calculator1Display.SelectOrientationVectors = 'None'
      calculator1Display.Scaling = 1
      calculator1Display.ScaleFactor = 0.003075522166277893
      calculator1Display.SelectScaleArray = 'Result'
      calculator1Display.GlyphType = 'Sphere'
      calculator1Display.GlyphTableIndexArray = 'Result'
      calculator1Display.GaussianRadius = 3e-05 #0.0015377610831389464
      calculator1Display.SetScaleArray = ['POINTS', 'Result']
      calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
      calculator1Display.OpacityArray = ['POINTS', 'Result']
      calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
      calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
      calculator1Display.PolarAxes = 'PolarAxesRepresentation'

      LoadPalette(paletteName='WhiteBackground')

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      calculator1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      calculator1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # get opacity transfer function/opacity map for 'Result'
      resultPWF = GetOpacityTransferFunction('Result')
      resultPWF.Points = [0.00013329293792503272, 0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]
      resultPWF.ScalarRangeInitialized = 1

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(calculator1)
      # ----------------------------------------------------------------
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'inputparticles': [1]}
  coprocessor.SetUpdateFrequencies(freqs)
  if requestSpecificArrays:
    arrays = [['cpu', 0], ['id', 0], ['velx', 0], ['vely', 0], ['velz', 0]]
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

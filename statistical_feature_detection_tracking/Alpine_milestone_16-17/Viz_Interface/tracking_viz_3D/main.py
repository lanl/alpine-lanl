## Post-hoc feature tracking algorithm development.
## Algorithm developed for ECP Alpine Project.
## This code is delivered as part of Alpine 16-17 P6 activity.
## Please contact Soumya Dutta (sdutta@lanl.gov) for any questions.
## Unauthorized use of this code is prohibited.

### Example to run: python main.py tstep featute_id
####################################################

import numpy as np
import sys 
import operator
import os
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
############################################################################


##################################################
## Main QT application class
class MainViewerApp(QtWidgets.QMainWindow):
    
    def __init__(self, lists,init_tstep):
        #Parent constructor
        super(MainViewerApp,self).__init__()
        self.vtk_widget = None
        self.ui = None
        self.setup(lists,init_tstep)
        
    def setup(self, lists,init_tstep):
        import volume_ui
        
        self.ui = volume_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.vtk_widget = QVolumeViewer(self.ui.vtk_panel, lists,init_tstep)
        self.ui.vtk_layout = QtWidgets.QHBoxLayout()
        self.ui.vtk_layout.addWidget(self.vtk_widget)
        self.ui.vtk_layout.setContentsMargins(0,0,0,0)
        self.ui.vtk_panel.setLayout(self.ui.vtk_layout)
        self.ui.pushButton_2.clicked.connect(self.vtk_widget.toggle_sim_volume)
        self.ui.pushButton_3.clicked.connect(self.vtk_widget.toggle_next)
        self.ui.pushButton_4.clicked.connect(self.vtk_widget.toggle_prev)
        self.ui.pushButton_5.clicked.connect(self.quit_app)

    def quit_app(self):
        QtCore.QCoreApplication.instance().quit()

    def initialize(self):
        self.vtk_widget.start()

##################################################
## Class controlling the volume rendering
class QVolumeViewer(QtWidgets.QFrame):

    def __init__(self, parent, lists, init_tstep):
        
        super(QVolumeViewer,self).__init__(parent)

        # Make tha actual QtWidget a child so that it can be re-parented
        self.interactor = QVTKRenderWindowInteractor(self)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.interactor)
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        self.show_sim_field = True

        def extract_idx(lst,idx): 
            return [item[idx] for item in lst] 

        self.all_tsteps = extract_idx(lists,0)
        self.all_fids = extract_idx(lists,1)
        self.all_featurefields = extract_idx(lists,2)
        self.all_simfields = extract_idx(lists,3)
        
        self.current_index = self.all_tsteps.index(init_tstep)
        self.max_index = len(self.all_tsteps)-1
        self.min_index = 0


        #####################################################################
        # Render the sim field volume
        #####################################################################
        fname = self.all_simfields[self.current_index]
        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName(fname)
        self.reader.Update()
        self.data = self.reader.GetOutput()
        self.data.GetPointData().SetScalars(self.reader.GetOutput().GetPointData().GetArray('feature_similarity'))

        # # Create transfer mapping
        # ## Scheme 1
        # opacityTransferFunction = vtk.vtkPiecewiseFunction()
        # opacityTransferFunction.AddPoint(0, 0.0)
        # opacityTransferFunction.AddPoint(0.929, 0.0)
        # opacityTransferFunction.AddPoint(0.93, 1.0)
        # opacityTransferFunction.AddPoint(1, 1.0)
        # colorTransferFunction = vtk.vtkColorTransferFunction()
        # colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.1)
        # colorTransferFunction.AddRGBPoint(0.5, 0.0, 0.0, 0.2)
        # colorTransferFunction.AddRGBPoint(1.0, 0.0, 0.0, 0.2)

        # Create transfer mapping
        ## Scheme 2
        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(0, 0.0)
        opacityTransferFunction.AddPoint(0.85, 0.0)
        opacityTransferFunction.AddPoint(0.90, 0.99)
        opacityTransferFunction.AddPoint(1, 1.0)
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(0.0, 0.2, 0.254902, 0.345098)
        colorTransferFunction.AddRGBPoint(0.981308, 0.517647, 0.615686, 0.7549)
        colorTransferFunction.AddRGBPoint(1.0, 0.890196, 0.956863, 0.984314)


        # The property describes how the data will look
        self.volumeProperty = vtk.vtkVolumeProperty()
        self.volumeProperty.SetColor(colorTransferFunction)
        self.volumeProperty.SetScalarOpacity(opacityTransferFunction)
        self.volumeProperty.ShadeOff()
        self.volumeProperty.SetInterpolationTypeToLinear()
        ## very important line, otherwise opacity func does not work as expected
        self.volumeProperty.SetScalarOpacityUnitDistance(0.005) 
        # The mapper / ray cast function know how to render the data
        self.volumeMapper = vtk.vtkSmartVolumeMapper()
        self.volumeMapper.SetBlendModeToComposite()
        self.volumeMapper.SetInputData(self.data)
        # The volume holds the mapper and the property and
        # can be used to position/orient the volume
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volumeMapper)
        self.volume.SetProperty(self.volumeProperty)
        self.volume.VisibilityOn()
        ######################################################


        #####################################################################
        # Render the individual feature
        #####################################################################
        fname = self.all_featurefields[self.current_index]
        self.reader1 = vtk.vtkXMLImageDataReader()
        self.reader1.SetFileName(fname)
        self.reader1.Update()
        self.data1 = self.reader1.GetOutput()
        self.data1.GetPointData().SetScalars(self.reader1.GetOutput().GetPointData().GetArray('feature_similarity'))

        # Create transfer mapping 
        ## Scheme 1
        opacityTransferFunction1 = vtk.vtkPiecewiseFunction()
        opacityTransferFunction1.AddPoint(0.0, 0.0)
        opacityTransferFunction1.AddPoint(0.5, 0.0)
        opacityTransferFunction1.AddPoint(1.0, 0.9)
        colorTransferFunction1 = vtk.vtkColorTransferFunction()
        colorTransferFunction1.AddRGBPoint(0.0, 0.0, 0.0, 0.501961)
        colorTransferFunction1.AddRGBPoint(0.88785, 1.0, 0.0, 0.0)
        colorTransferFunction1.AddRGBPoint(1.0, 1.0, 1.0, 0.0)


        # The property describes how the data will look
        self.volumeProperty1 = vtk.vtkVolumeProperty()
        self.volumeProperty1.SetColor(colorTransferFunction1)
        self.volumeProperty1.SetScalarOpacity(opacityTransferFunction1)
        self.volumeProperty1.ShadeOff()
        self.volumeProperty1.SetInterpolationTypeToLinear()
        ## very important line, otherwise opacity func does not work as expected
        self.volumeProperty1.SetScalarOpacityUnitDistance(0.001) 
        # The mapper / ray cast function know how to render the data
        self.volumeMapper1 = vtk.vtkSmartVolumeMapper()
        self.volumeMapper1.SetBlendModeToComposite()
        self.volumeMapper1.SetInputData(self.data1)
        # The volume holds the mapper and the property and
        # can be used to position/orient the volume
        self.volume1 = vtk.vtkVolume()
        self.volume1.SetMapper(self.volumeMapper1)
        self.volume1.SetProperty(self.volumeProperty1)
        self.volume1.VisibilityOn()
        ######################################################


        #####################
        ## Outline
        self.outline = vtk.vtkOutlineFilter()
        self.outline.SetInputData(self.data)
        self.outline_mapper = vtk.vtkPolyDataMapper()
        self.outline_mapper.SetInputConnection(self.outline.GetOutputPort())
        self.outline_actor = vtk.vtkActor()
        self.outline_actor.SetMapper(self.outline_mapper)
        self.outline_actor.GetProperty().SetColor(0,0,0)
        #self.outline_actor.GetProperty().SetLineWidth(2.0)


        # create a text actor
        self.txt = vtk.vtkTextActor()
        self.txt.SetInput('Time step: ' + str(init_tstep*100))
        self.txt.SetDisplayPosition(10,590)
        txtprop=self.txt.GetTextProperty()
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(16)
        txtprop.SetColor(0,0,0)
        
        txt1 = vtk.vtkTextActor()
        txt1.SetInput('MFiX-Exa Bubble Tracking')
        txtprop1=txt1.GetTextProperty()
        txtprop1.SetFontFamilyToArial()
        txtprop1.SetFontSize(16)
        txtprop1.SetColor(0,0,0)
        txt1.SetDisplayPosition(100,10)

        ######################################################
        # Setup VTK environment
        self.renderer = vtk.vtkRenderer()

        ## Add all the volumes to renderer
        ## The order of adding volumes matter here. 
        ## The later added volumes will override the previously added ones
        self.renderer.AddVolume(self.volume)
        self.renderer.AddVolume(self.volume1)
        self.renderer.AddActor(self.outline_actor)
        self.renderer.AddActor(self.txt)
        self.renderer.AddActor(txt1)

        self.render_window = self.interactor.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.render_window.SetInteractor(self.interactor)
        self.renderer.SetBackground(1,1,1)


    #######################################
    def start(self):
        self.interactor.Initialize()
        self.interactor.Start()

    ## Callback for pushButton_2
    def toggle_sim_volume(self):
        self.show_sim_field = not self.show_sim_field
        if self.show_sim_field:
            self.volume.VisibilityOn()
        else:
            self.volume.VisibilityOff()
        self.render_window.Render()

    ## Callback for pushButton_3
    def toggle_next(self):
        #print ('showing next time step')
        if (self.current_index + 1 <= self.max_index):

            self.current_index = self.current_index + 1
            
            ## update sim vol
            fname = self.all_simfields[self.current_index]
            self.reader.SetFileName(fname)
            self.reader.Update()
            self.data = self.reader.GetOutput()
            self.data.GetPointData().SetScalars(self.reader.GetOutput().GetPointData().GetArray('feature_similarity'))
            self.volumeMapper.SetInputData(self.data)

            ## upadte feature field
            fname = self.all_featurefields[self.current_index]
            self.reader1.SetFileName(fname)
            self.reader1.Update()
            self.data1 = self.reader1.GetOutput()
            self.data1.GetPointData().SetScalars(self.reader1.GetOutput().GetPointData().GetArray('feature_similarity'))
            self.volumeMapper1.SetInputData(self.data1)

            ## change text for showing timestep
            curr_tstep = self.all_tsteps[self.current_index]
            self.txt.SetInput('Time step: ' + str(curr_tstep*100))
            
        self.render_window.Render()

    ## Callback for pushButton_4
    def toggle_prev(self):
        #print ('showing previous time step')
        if (self.current_index -1 >= self.min_index):

            self.current_index = self.current_index - 1
            
            ## update sim vol
            fname = self.all_simfields[self.current_index]
            self.reader.SetFileName(fname)
            self.reader.Update()
            self.data = self.reader.GetOutput()
            self.data.GetPointData().SetScalars(self.reader.GetOutput().GetPointData().GetArray('feature_similarity'))
            self.volumeMapper.SetInputData(self.data)

            ## upadte feature field
            fname = self.all_featurefields[self.current_index]
            self.reader1.SetFileName(fname)
            self.reader1.Update()
            self.data1 = self.reader1.GetOutput()
            self.data1.GetPointData().SetScalars(self.reader1.GetOutput().GetPointData().GetArray('feature_similarity'))
            self.volumeMapper1.SetInputData(self.data1)

            ## change text for showing timestep
            curr_tstep = self.all_tsteps[self.current_index]
            self.txt.SetInput('Time step: ' + str(curr_tstep*100))

        self.render_window.Render()

#######################################################################

if __name__ == "__main__":

    ## read input params
    init_tstep = int(sys.argv[1])
    init_fid = int(sys.argv[2])

    src_path = '/Users/sdutta/Codes/Alpine_milestone_16-17/Visualization_interface_codes/mfix_tracking_viz_3D/'
    sim_field_path = '/Users/sdutta/Codes/statistical_data_comparison/out/mfix_local_grid/'
    path_to_all_features = '/Users/sdutta/Codes/Alpine_milestone_16-17/out/feature_tracking_cdb/bubble_tracked' \
                            '_' + str(init_tstep) + '_' + str(init_fid) + '.cdb/images/'

    tsteps = []
    fids = []
    lists = []
    for file in sorted(os.listdir(path_to_all_features)):
        if file.endswith(".vti"):
            fullname = os.path.join(path_to_all_features,file)
            fname = os.path.splitext(file)[0]
            ffname = fname.split('_')
            idx = ffname[2]
            tstep = ffname[3]

            tsteps.append(int(tstep))
            fids.append(int(idx))

            sim_field_name = sim_field_path + 'slic_compare_' + str(tstep) + '.vti'

            lists.append([int(tstep),int(idx),fullname,sim_field_name])

    ## sort the list based on time steps
    lists = sorted(lists, key = operator.itemgetter(0))
   
    # Recompile ui
    with open(src_path + 'volume_view.ui') as ui_file:
        with open(src_path + 'volume_ui.py',"w") as py_ui_file:
            uic.compileUi(ui_file,py_ui_file)

    ## Create an App instance
    app = QtWidgets.QApplication(sys.argv)

    # Use a palette to switch to dark colors:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(75, 75, 75))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    # Launch feature tracking application
    main_window = MainViewerApp(lists,init_tstep)
    main_window.show()
    main_window.initialize()
    app.exec_()


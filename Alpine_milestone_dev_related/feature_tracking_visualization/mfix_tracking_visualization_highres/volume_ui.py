# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/sdutta/Codes/tracking_interface/mfix_tracking_visualization_highres/volume_view.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1164, 658)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.frame = QtWidgets.QFrame(self.splitter)
        self.frame.setMaximumSize(QtCore.QSize(414, 16777215))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.formLayout = QtWidgets.QFormLayout(self.frame)
        self.formLayout.setObjectName("formLayout")
        self.pushButton_4 = QtWidgets.QPushButton(self.frame)
        self.pushButton_4.setObjectName("pushButton_4")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.pushButton_4)
        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setObjectName("pushButton_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.pushButton_3)
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setObjectName("pushButton_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setObjectName("pushButton")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.pushButton)
        self.pushButton_5 = QtWidgets.QPushButton(self.frame)
        self.pushButton_5.setObjectName("pushButton_5")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.pushButton_5)
        self.vtk_panel = QtWidgets.QFrame(self.splitter)
        self.vtk_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.vtk_panel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.vtk_panel.setObjectName("vtk_panel")
        self.horizontalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1164, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Bubble Tracking Visualization"))
        self.pushButton_4.setText(_translate("MainWindow", "Track Previous"))
        self.pushButton_3.setText(_translate("MainWindow", "Track Next"))
        self.pushButton_2.setText(_translate("MainWindow", "Toggle Similarity Field"))
        self.pushButton.setText(_translate("MainWindow", "Toggle Velocity Field"))
        self.pushButton_5.setText(_translate("MainWindow", "Quit"))
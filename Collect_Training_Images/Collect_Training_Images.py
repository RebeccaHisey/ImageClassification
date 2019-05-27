import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import subprocess

#
# Collect_Training_Images
#

class Collect_Training_Images(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Collect_Training_Images" # TODO make this more human readable by adding spaces
    self.parent.categories = ["ImageClassification"]
    self.parent.dependencies = []
    self.parent.contributors = ["Rebecca Hisey (Perk Lab), Tamas Ungi (Perk Lab)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is a module to collect training images for the purpose of training a convolutional neural network for image classification"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# Collect_Training_ImagesWidget
#

class Collect_Training_ImagesWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = Collect_Training_ImagesLogic()

    self.moduleDir = os.path.dirname(slicer.modules.collect_training_images.path)

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #self.imageSaveDirectory = qt.QLineEdit("Select directory to save images")
    #parametersFormLayout.addRow(self.imageSaveDirectory)

    self.modelSelector = qt.QComboBox()
    self.modelSelector.addItems(["Select model"])
    modelDirectoryContents = os.listdir(os.path.join(self.moduleDir,os.pardir,"Models"))
    modelNames = [dir for dir in modelDirectoryContents if dir.find(".") == -1 and dir != "Dockerfile"]
    self.modelSelector.addItems(["Create new model"])
    self.modelSelector.addItems(modelNames)
    parametersFormLayout.addRow(self.modelSelector)

    #self.imageSaveDirectoryLabel = qt.QLabel("Training photo directory:")
    #parametersFormLayout.addRow(self.imageSaveDirectoryLabel)

    self.imageSaveDirectoryLineEdit = ctk.ctkPathLineEdit()
    #node = self.logic.getParameterNode()
    imageSaveDirectory = os.path.dirname(slicer.modules.collect_training_images.path)
    self.imageSaveDirectoryLineEdit.currentPath = imageSaveDirectory
    self.imageSaveDirectoryLineEdit.filters = ctk.ctkPathLineEdit.Dirs
    self.imageSaveDirectoryLineEdit.options = ctk.ctkPathLineEdit.DontUseSheet
    self.imageSaveDirectoryLineEdit.options = ctk.ctkPathLineEdit.ShowDirsOnly
    self.imageSaveDirectoryLineEdit.showHistoryButton = False
    self.imageSaveDirectoryLineEdit.setMinimumWidth(100)
    self.imageSaveDirectoryLineEdit.setMaximumWidth(500)
    #parametersFormLayout.addRow(self.imageSaveDirectoryLineEdit)

    self.imageClassComboBox = qt.QComboBox()
    self.imageClassComboBox.addItems(["Select image class","Create new image class"])

    parametersFormLayout.addRow(self.imageClassComboBox)

    #
    # Start/Stop Image Collection Button
    #
    self.startStopCollectingImagesButton = qt.QPushButton("Start Image Collection")
    self.startStopCollectingImagesButton.toolTip = "Collect training images."
    self.startStopCollectingImagesButton.enabled = False
    parametersFormLayout.addRow(self.startStopCollectingImagesButton)


    self.infoLabel = qt.QLabel("")
    parametersFormLayout.addRow(self.infoLabel)

    # connections
    self.modelSelector.connect('currentIndexChanged(int)',self.onModelSelected)
    self.startStopCollectingImagesButton.connect('clicked(bool)', self.onStartStopCollectingImagesButton)
    self.imageClassComboBox.connect('currentIndexChanged(int)',self.onImageClassSelected)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Start/Stop Collecting Images Button state
    self.onSelect()
    try:
      self.webcamReference = slicer.util.getNode('Webcam_Reference')
    except slicer.util.MRMLNodeNotFoundException:
    #if not self.webcamReference:
      imageSpacing = [0.2, 0.2, 0.2]
      imageData = vtk.vtkImageData()
      imageData.SetDimensions(640, 480, 1)
      imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
      thresholder = vtk.vtkImageThreshold()
      thresholder.SetInputData(imageData)
      thresholder.SetInValue(0)
      thresholder.SetOutValue(0)
      # Create volume node
      self.webcamReference = slicer.vtkMRMLVectorVolumeNode()
      self.webcamReference.SetName('Webcam_Reference')
      self.webcamReference.SetSpacing(imageSpacing)
      self.webcamReference.SetImageDataConnection(thresholder.GetOutputPort())
      # Add volume to scene
      slicer.mrmlScene.AddNode(self.webcamReference)
      displayNode = slicer.vtkMRMLVectorVolumeDisplayNode()
      slicer.mrmlScene.AddNode(displayNode)
      self.webcamReference.SetAndObserveDisplayNodeID(displayNode.GetID())

    self.webcamConnectorNode = self.createWebcamPlusConnector()
    self.webcamConnectorNode.Start()
    self.setupWebcamResliceDriver()

  def createWebcamPlusConnector(self):
    try:
      webcamConnectorNode = slicer.util.getNode('WebcamPlusConnector')
    except slicer.util.MRMLNodeNotFoundException:
    #if not webcamConnectorNode:
      webcamConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      #webcamConnectorNode.SetLogErrorIfServerConnectionFailed(False)
      webcamConnectorNode.SetName('WebcamPlusConnector')
      slicer.mrmlScene.AddNode(webcamConnectorNode)
      #hostNamePort = self.parameterNode.GetParameter('PlusWebcamServerHostNamePort')
      hostNamePort = "localhost:18944"
      [hostName, port] = hostNamePort.split(':')
      webcamConnectorNode.SetTypeClient(hostName, int(port))
      logging.debug('Webcam PlusConnector Created')
    return webcamConnectorNode

  def setupWebcamResliceDriver(self):
    # Setup the volume reslice driver for the webcam.
    self.webcamReference = slicer.util.getNode('Webcam_Reference')

    layoutManager = slicer.app.layoutManager()
    yellowSlice = layoutManager.sliceWidget('Yellow')
    yellowSliceLogic = yellowSlice.sliceLogic()
    yellowSliceLogic.GetSliceCompositeNode().SetBackgroundVolumeID(self.webcamReference.GetID())

    resliceLogic = slicer.modules.volumereslicedriver.logic()
    if resliceLogic:
      yellowNode = slicer.util.getNode('vtkMRMLSliceNodeYellow')
      yellowNode.SetSliceResolutionMode(slicer.vtkMRMLSliceNode.SliceResolutionMatchVolumes)
      resliceLogic.SetDriverForSlice(self.webcamReference.GetID(), yellowNode)
      resliceLogic.SetModeForSlice(6, yellowNode)
      resliceLogic.SetFlipForSlice(False, yellowNode)
      resliceLogic.SetRotationForSlice(180, yellowNode)
      yellowSliceLogic.FitSliceToAll()

  def cleanup(self):
    pass

  def openCreateNewModelWindow(self):
    self.createNewModelWidget = qt.QDialog()
    self.createNewModelWidget.setModal(True)
    self.createNewModelFrame = qt.QFrame(self.createNewModelWidget)
    self.createNewModelFrame.setFrameStyle(0x0006)
    self.createNewModelWidget.setWindowTitle('Create New Model')
    self.createNewModelPopupGeometry = qt.QRect()
    mainWindow = slicer.util.mainWindow()
    if mainWindow:
      mainWindowGeometry = mainWindow.geometry
      self.windowWidth = mainWindow.width * 0.35
      self.windowHeight = mainWindow.height * 0.35
      self.createNewModelPopupGeometry.setWidth(self.windowWidth)
      self.createNewModelPopupGeometry.setHeight(self.windowHeight)
      self.popupPositioned = False
      self.createNewModelWidget.setGeometry(self.createNewModelPopupGeometry)
      self.createNewModelFrame.setGeometry(self.createNewModelPopupGeometry)
      self.createNewModelWidget.move(mainWindow.width / 2.0 - self.windowWidth,
                                     mainWindow.height / 2 - self.windowHeight)
    self.createNewModelLayout = qt.QVBoxLayout()
    self.createNewModelLayout.setContentsMargins(12, 4, 4, 4)
    self.createNewModelLayout.setSpacing(4)

    self.createNewModelButtonLayout = qt.QFormLayout()
    self.createNewModelButtonLayout.setContentsMargins(12, 4, 4, 4)
    self.createNewModelButtonLayout.setSpacing(4)

    self.modelNameLineEdit = qt.QLineEdit("Model Name")
    self.createNewModelButtonLayout.addRow(self.modelNameLineEdit)

    self.createNewModelButton = qt.QPushButton("Add Model")
    self.createNewModelButtonLayout.addRow(self.createNewModelButton)

    self.errorLabel = qt.QLabel("")
    self.createNewModelButtonLayout.addRow(self.errorLabel)

    self.createNewModelButton.connect('clicked(bool)', self.onNewModelAdded)

    self.createNewModelLayout.addLayout(self.createNewModelButtonLayout)
    self.createNewModelFrame.setLayout(self.createNewModelLayout)

  def openCreateNewImageClassWindow(self):
    self.createNewImageClassWidget = qt.QDialog()
    self.createNewImageClassWidget.setModal(True)
    self.createNewImageClassFrame = qt.QFrame(self.createNewImageClassWidget)
    self.createNewImageClassFrame.setFrameStyle(0x0006)
    self.createNewImageClassWidget.setWindowTitle('Create New Model')
    self.createNewImageClassPopupGeometry = qt.QRect()
    mainWindow = slicer.util.mainWindow()
    if mainWindow:
      mainWindowGeometry = mainWindow.geometry
      self.windowWidth = mainWindow.width * 0.35
      self.windowHeight = mainWindow.height * 0.35
      self.createNewImageClassPopupGeometry.setWidth(self.windowWidth)
      self.createNewImageClassPopupGeometry.setHeight(self.windowHeight)
      self.popupPositioned = False
      self.createNewImageClassWidget.setGeometry(self.createNewImageClassPopupGeometry)
      self.createNewImageClassFrame.setGeometry(self.createNewImageClassPopupGeometry)
      self.createNewImageClassWidget.move(mainWindow.width / 2.0 - self.windowWidth,
                                     mainWindow.height / 2 - self.windowHeight)
    self.createNewImageClassLayout = qt.QVBoxLayout()
    self.createNewImageClassLayout.setContentsMargins(12, 4, 4, 4)
    self.createNewImageClassLayout.setSpacing(4)

    self.createNewImageClassButtonLayout = qt.QFormLayout()
    self.createNewImageClassButtonLayout.setContentsMargins(12, 4, 4, 4)
    self.createNewImageClassButtonLayout.setSpacing(4)

    self.imageClassNameLineEdit = qt.QLineEdit("Image Class Name")
    self.createNewImageClassButtonLayout.addRow(self.imageClassNameLineEdit)

    self.createNewImageClassButton = qt.QPushButton("Add Image Class")
    self.createNewImageClassButtonLayout.addRow(self.createNewImageClassButton)

    self.imageErrorLabel = qt.QLabel("")
    self.createNewImageClassButtonLayout.addRow(self.imageErrorLabel)

    self.createNewImageClassButton.connect('clicked(bool)', self.onNewImageClassAdded)

    self.createNewImageClassLayout.addLayout(self.createNewImageClassButtonLayout)
    self.createNewImageClassFrame.setLayout(self.createNewImageClassLayout)

  def onModelSelected(self):
    if self.modelSelector.currentText == "Create new model":
      try:
        self.createNewModelWidget.show()
      except AttributeError:
        self.openCreateNewModelWindow()
        self.createNewModelWidget.show()
    elif self.modelSelector.currentText != "Select model":
      self.currentModelName = self.modelSelector.currentText
      self.trainingPhotoPath = os.path.join(self.moduleDir,os.pardir,"Models",self.modelSelector.currentText,"training_photos")
      self.imageSaveDirectoryLineEdit.currentPath = self.trainingPhotoPath
      self.addImageClassesToComboBox()
    else:
      for i in range(2, self.imageClassComboBox.count + 1):
        self.imageClassComboBox.removeItem(i)


  def addImageClassesToComboBox(self):
    for i in range(2,self.imageClassComboBox.count + 1):
      self.imageClassComboBox.removeItem(i)
    imageClassList = os.listdir(self.trainingPhotoPath)
    self.imageClassList = [dir for dir in imageClassList if dir.rfind(".") == -1] #get only directories
    self.imageClassComboBox.addItems(self.imageClassList)

  def onNewModelAdded(self):
    self.currentModelName = self.modelNameLineEdit.text
    try:
      modelPath = os.path.join(self.moduleDir,os.pardir,"Models",self.currentModelName)
      os.mkdir(modelPath)
      os.mkdir(os.path.join(modelPath,"training_photos"))
      os.mkdir(os.path.join(modelPath,"trained_model"))
      classifierPath = os.path.join(self.moduleDir,os.pardir,"Models/classifierContainer/trained_model",self.currentModelName)
      os.makedirs(classifierPath)
      self.modelSelector.addItems([self.currentModelName])
      modelIndex = self.modelSelector.findText(self.currentModelName)
      self.modelSelector.currentIndex = modelIndex
      self.createNewModelWidget.hide()
      self.modelNameLineEdit.setText("Model Name")
      self.errorLabel.setText("")
    except WindowsError:
      self.modelNameLineEdit.setText("Model Name")
      self.errorLabel.setText("A model with the name " + self.currentModelName + " already exists")


  def onNewImageClassAdded(self):
    self.currentImageClassName = self.imageClassNameLineEdit.text
    try:
      imageClassPath = os.path.join(self.trainingPhotoPath,self.currentImageClassName)
      os.mkdir(imageClassPath)
      self.imageClassComboBox.addItems([self.currentImageClassName])
      imageClassIndex = self.imageClassComboBox.findText(self.currentImageClassName)
      self.imageClassComboBox.currentIndex = imageClassIndex
      self.createNewImageClassWidget.hide()
      self.imageClassNameLineEdit.setText("Image Class Name")
      self.imageErrorLabel.setText("")
    except WindowsError:
      self.imageClassNameLineEdit.setText("Image Class Name")
      self.imageErrorLabel.setText("An image class with the name " + self.currentImageClassName + " already exists")

  def onImageClassSelected(self):
    if self.imageClassComboBox.currentText == "Create new image class":
      try:
        self.createNewImageClassWidget.show()
      except AttributeError:
        self.openCreateNewImageClassWindow()
        self.createNewImageClassWidget.show()
    elif self.imageClassComboBox.currentText != "Select image class":
      self.currentImageClassName = self.imageClassComboBox.currentText
      self.currentImageClassFilePath = os.path.join(self.trainingPhotoPath,self.currentImageClassName)
      self.startStopCollectingImagesButton.enabled = True

  def onSelect(self):
    self.startStopCollectingImagesButton.enabled =  self.imageClassComboBox.currentText!= "Select image class" and self.imageClassComboBox.currentText!= "Create new image class"


  def onStartStopCollectingImagesButton(self):

    if self.startStopCollectingImagesButton.text == "Start Image Collection":
      self.collectingImages = False
      self.startStopCollectingImagesButton.setText("Stop Image Collection")
    else:
      self.collectingImages = True
      self.startStopCollectingImagesButton.setText("Start Image Collection")
    self.logic.startImageCollection(self.collectingImages, self.currentImageClassName,self.currentImageClassFilePath)

  def onRetrainClicked(self):
    self.infoLabel.setText("Retraining model, this may take up to 30min\nNavigate to localhost:6006 in browser to visualize training")
    self.logic.retrainClassifier(self.modelSelector.currentText)


#
# Collect_Training_ImagesLogic
#

class Collect_Training_ImagesLogic(ScriptedLoadableModuleLogic):
  def startImageCollection(self,imageCollectionStarted,imageClassName, imageClassFilePath):
    try:
      # the module is in the python path
      import cv2
    except ImportError:
      # for the build directory, load from the file
      import imp, platform
      if platform.system() == 'Windows':
        cv2File = 'cv2.pyd'
        cv2Path = '../../../../OpenCV-build/lib/Release/' + cv2File
      else:
        cv2File = 'cv2.so'
        cv2Path = '../../../../OpenCV-build/lib/' + cv2File
      scriptPath = os.path.dirname(os.path.abspath(__file__))
      cv2Path = os.path.abspath(os.path.join(scriptPath, cv2Path))
      # in the build directory, this path should exist, but in the installed extension
      # it should be in the python pat, so only use the short file name
      if not os.path.isfile(cv2Path):
        cv2Path = cv2File
      cv2 = imp.load_dynamic('cv2', cv2File)
    self.collectingImages = imageCollectionStarted
    self.imageClassName = imageClassName
    self.imageClassFilePath = imageClassFilePath
    if self.collectingImages == False:
      self.webcamImageVolume = slicer.util.getNode('Webcam_Reference')
      self.webcamImageObserver = self.webcamImageVolume.AddObserver(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, self.onStartCollectingImages)
      logging.info("Start collecting images")
    else:
      self.webcamImageVolume = slicer.util.getNode('Webcam_Reference')
      self.webcamImageVolume.RemoveObserver(self.webcamImageObserver)
      self.webcamImageObserver = None
      self.numImagesInFile = len(os.listdir(self.imageClassFilePath))
      logging.info("Saved " + str(self.numImagesInFile) + " to directory : " + str(self.imageClassFilePath))

  def onStartCollectingImages(self,caller,eventID):
    import numpy
    try:
      # the module is in the python path
      import cv2
    except ModuleNotFoundError:
      # for the build directory, load from the file
      import imp, platform
      if platform.system() == 'Windows':
        cv2File = 'cv2.pyd'
        cv2Path = '../../../../OpenCV-build/lib/Release/' + cv2File
      else:
        cv2File = 'cv2.so'
        cv2Path = '../../../../OpenCV-build/lib/' + cv2File
      scriptPath = os.path.dirname(os.path.abspath(__file__))
      cv2Path = os.path.abspath(os.path.join(scriptPath, cv2Path))
      # in the build directory, this path should exist, but in the installed extension
      # it should be in the python pat, so only use the short file name
      if not os.path.isfile(cv2Path):
        cv2Path = cv2File
      cv2 = imp.load_dynamic('cv2', cv2File)

    # Get the vtkImageData as an np.array.
    self.numImagesInFile = len(os.listdir(self.imageClassFilePath))
    logging.info(self.numImagesInFile)
    imData = self.getVtkImageDataAsOpenCVMat('Webcam_Reference')
    imDataBGR = cv2.cvtColor(imData,cv2.COLOR_RGB2BGR)
    if self.numImagesInFile < 10:
      fileName = self.imageClassName + "_0" + str(self.numImagesInFile) + ".jpg"
    else:
      fileName = self.imageClassName + "_" + str(self.numImagesInFile) + ".jpg"
    cv2.imwrite(os.path.join(self.imageClassFilePath,fileName),imDataBGR)

  def getVtkImageDataAsOpenCVMat(self, volumeNodeName):
    cameraVolume = slicer.util.getNode(volumeNodeName)
    image = cameraVolume.GetImageData()
    shape = list(cameraVolume.GetImageData().GetDimensions())
    shape.reverse()
    components = image.GetNumberOfScalarComponents()
    if components > 1:
      shape.append(components)
      shape.remove(1)
    imageMat = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)
    return imageMat

  def retrainClassifier(self,modelName):
    logging.info("creating docker container")
    retrainContainerPath = slicer.modules.collect_training_images.path
    ExtensionPath = retrainContainerPath.replace("Collect_Training_Images/Collect_Training_Images.py","")
    retrainContainerPath = retrainContainerPath.replace("Collect_Training_Images/Collect_Training_Images.py","Models")
    volumeflag = "-v=" + retrainContainerPath + ":/app"
    modelNameFlag = str("MODELNAME=" + modelName)
    numTrainingStepsFlag = "NUMTRAININGSTEPS=" + str(6000)
    trainingBatchSize="TRAINBATCHSIZE=" + str(150)
    logging.info(modelNameFlag)
    #cmd = ["C:/Program Files/Docker/Docker/resources/bin/docker.exe", "run", "-i", "--name", "retrain", "--rm",
    #       volumeflag, "-e", modelNameFlag, "-e", numTrainingStepsFlag, "-e", trainingBatchSize, "-p", "80:5000", "-p","6006:6006","retrainimage"]
    cmd = ["call",ExtensionPath+"\StartRetrain",ExtensionPath,modelName,"100","50"]
    p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
    logging.info("starting docker container")
    [output,error] = p.communicate()
    print (output)
    print(error)
    #logging.info(error)
    #cmd = ["C:/Program Files/Docker/Docker/resources/bin/docker.exe", "start", "-i", "retrain"]
    #q = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout = subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
    #[output,error] = q.communicate()





class Collect_Training_ImagesTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)
import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import subprocess
import numpy
import time

#
# CNN_Image_Classifier
#

class CNN_Image_Classifier(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "CNN_Image_Classifier" # TODO make this more human readable by adding spaces
    self.parent.categories = ["ImageClassification"]
    self.parent.dependencies = []
    self.parent.contributors = ["Rebecca Hisey (Perk Lab)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This extension allows for the training and use of a convolutional neural network (inception v3) to classify images 
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
""" # replace with organization, grant and thanks.

#
# CNN_Image_ClassifierWidget
#

class CNN_Image_ClassifierWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = CNN_Image_ClassifierLogic()
    self.moduleDir = os.path.dirname(slicer.modules.collect_training_images.path)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    self.modelSelector = qt.QComboBox()
    self.modelSelector.addItems(["Select model"])
    modelDirectoryContents = os.listdir(os.path.join(self.moduleDir, os.pardir, "Models/retrainContainer"))
    modelNames = [dir for dir in modelDirectoryContents if dir.find(".") == -1 and dir != "Dockerfile"]
    self.modelSelector.addItems(["Create new model"])
    self.modelSelector.addItems(modelNames)
    parametersFormLayout.addRow(self.modelSelector)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Start")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    #
    # Object table
    #
    self.objectTable = qt.QTableWidget()
    self.objectTable.setColumnCount(3)
    self.objectTable.setHorizontalHeaderLabels(["Name","Found","Confidence"])
    parametersFormLayout.addRow(self.objectTable)



    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.modelSelector.connect('currentIndexChanged(int)',self.onModelSelected)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

    self.webcamReference = slicer.util.getNode('Webcam_Reference')
    if not self.webcamReference:
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
    webcamConnectorNode = slicer.util.getNode('WebcamPlusConnector')
    if not webcamConnectorNode:
      webcamConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      webcamConnectorNode.SetLogErrorIfServerConnectionFailed(False)
      webcamConnectorNode.SetName('WebcamPlusConnector')
      slicer.mrmlScene.AddNode(webcamConnectorNode)
      # hostNamePort = self.parameterNode.GetParameter('PlusWebcamServerHostNamePort')
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

  def onSelect(self):
    self.applyButton.enabled = True

  def onApplyButton(self):
    if self.applyButton.text == "Start":
      self.logic.run(self.objectTable)
      self.applyButton.setText("Stop")
    else:
      self.logic.stopClassifier()
      self.applyButton.setText("Start")

  def onModelSelected(self):
    if self.modelSelector.currentText != "Create new model" and self.modelSelector.currentText != "Select model":
      self.currentModelDirectory = os.path.join(self.moduleDir, os.pardir, "Models/retrainContainer", self.modelSelector.currentText)
      modelObjectClasses = os.listdir(os.path.join(self.currentModelDirectory,"training_photos"))
      self.currentObjectClasses = [dir for dir in modelObjectClasses if dir.find(".") == -1]
      self.objectTable.setRowCount(len(self.currentObjectClasses))
      for i in range (len(self.currentObjectClasses)):
        self.objectTable.setItem(i,0,qt.QTableWidgetItem(self.currentObjectClasses[i]))
        self.objectTable.setItem(i,1,qt.QTableWidgetItem("No"))

#
# CNN_Image_ClassifierLogic
#

class CNN_Image_ClassifierLogic(ScriptedLoadableModuleLogic):

  def run(self,objectTable):
    self.objectTable = objectTable
    self.numObjects = self.objectTable.rowCount
    self.webcamReference = slicer.util.getNode('Webcam_Reference')
    numpy.set_printoptions(threshold=numpy.nan)
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

    classifierContainerPath = slicer.modules.collect_training_images.path
    self.classifierContainerPath = classifierContainerPath.replace("Collect_Training_Images/Collect_Training_Images.py",
                                                        "Models/classifierContainer")
    volumeflag = "-v=" + self.classifierContainerPath.replace("C:","/c") + ":/app"
    cmd = ["C:/Program Files/Docker/Docker/resources/bin/docker.exe", "create","--name", "classify",
						   volumeflag,"-p", "80:5000", "classifierimage"]
    p = subprocess.Popen(cmd, stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True)
    p.communicate()
    cmd = ["C:/Program Files/Docker/Docker/resources/bin/docker.exe", "start", "classify"]
    p = subprocess.call(cmd, stdin=subprocess.PIPE,shell=True)
    time.sleep(1.5)
    self.currentLabel = ""
    self.lastUpdateSec = 0
    self.webcamObserver = self.webcamReference.AddObserver(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent,self.onWebcamImageModified)

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

  def onWebcamImageModified(self,caller, eventID):
    if time.time() - self.lastUpdateSec > 1.5:
      self.confidences = []
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
      cmd = ["C:/Program Files/Docker/Docker/resources/bin/docker.exe", "container", "pause",
             "classify"]
      p = subprocess.call(cmd, shell=True)
      imdata = self.getVtkImageDataAsOpenCVMat('Webcam_Reference')
      imDataBGR = cv2.cvtColor(imdata, cv2.COLOR_RGB2BGR)
      #img = cv2.imread(image)
      imgSize = imDataBGR.shape
      imgSize = numpy.array(imgSize)
      start = time.time()
      numpy.save(self.classifierContainerPath + '/pictureSize.npy',imgSize)
      numpy.save(self.classifierContainerPath + '/picture.npy',imDataBGR)
      fileMod1 = os.path.getmtime(self.classifierContainerPath + '/textLabels.txt')
      cmd = ["C:/Program Files/Docker/Docker/resources/bin/docker.exe", "container", "unpause",
             "classify"]
      p = subprocess.call(cmd, shell=True)
      currentTime = time.time()
      fileRead = False
      fileMod2 = os.path.getmtime(self.classifierContainerPath + '/textLabels.txt')
      while fileMod2 == fileMod1 and time.time() - start < 5:
          time.sleep(0.2)
          fileMod2 = os.path.getmtime(self.classifierContainerPath + '/textLabels.txt')
      cmd = ["C:/Program Files/Docker/Docker/resources/bin/docker.exe", "container", "pause",
             "classify"]
      p = subprocess.call(cmd, shell=True)
      fileMod3 = os.path.getmtime(self.classifierContainerPath + '/textLabels.txt')
      while not fileRead and time.time() - currentTime < 7:
          try:
              fName = self.classifierContainerPath + '/textLabels.txt'
              with open(fName) as f:
                  self.currentLabel = f.readline()
                  for i in range(self.numObjects):
                    self.confidences.append(f.readline())
              fileRead = True
          except IOError:
              time.sleep(0.05)
      end = time.time()
      #logging.info(self.currentLabel == "sunglasses\n")
      if self.currentLabel == 'sunglasses\n' and float(self.confidences[0]) < 0.60:
        self.currentLabel = "nothing"
      elif self.currentLabel == 'watch\n' and float(self.confidences[1]) < 0.85:
        self.currentLabel = "nothing"
      logging.info(self.currentLabel + ' ' + self.confidences[0] + " " + self.confidences[1])
      logging.info(self.objectTable.item(1,0).text())
      self.lastUpdateSec = time.time()
      self.updateObjectTable()

  def stopClassifier(self):
    self.webcamReference.RemoveObserver(self.webcamObserver)
    self.webcamObserver = None
    cmd = ["C:/Program Files/Docker/Docker/resources/bin/docker.exe", "container", "stop", "classify"]
    p = subprocess.call(cmd, shell=True)
    cmd = ["C:/Program Files/Docker/Docker/resources/bin/docker.exe", "container", "rm", "classify"]
    p = subprocess.call(cmd, shell=True)

  def updateObjectTable(self):
    for i in range(self.numObjects):
      if self.currentLabel == str(self.objectTable.item(i,0).text() + "\n"):
        self.objectTable.setItem(i,1,qt.QTableWidgetItem("Yes"))
        self.objectTable.setItem(i,2,qt.QTableWidgetItem(self.confidences[i]))
      else:
        self.objectTable.setItem(i, 1, qt.QTableWidgetItem("No"))
        self.objectTable.setItem(i, 2, qt.QTableWidgetItem(self.confidences[i]))


class CNN_Image_ClassifierTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_CNN_Image_Classifier1()

  def test_CNN_Image_Classifier1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = CNN_Image_ClassifierLogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')

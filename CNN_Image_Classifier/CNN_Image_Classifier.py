from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import subprocess
import itertools
import logging
import subprocess
import numpy as np
import time

import tensorflow as tf


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
    modelDirectoryContents = os.listdir(os.path.join(self.moduleDir, os.pardir, "Models"))
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

    '''
    self.classifyAllFramesButton = qt.QPushButton("Classify all frames")
    self.classifyAllFramesButton.toolTip = "classify all frames in a sequence"
    parametersFormLayout.addRow(self.classifyAllFramesButton)
    '''

    #
    # Object table
    #
    self.objectTable = qt.QTableWidget()
    self.objectTable.setColumnCount(3)
    self.objectTable.setHorizontalHeaderLabels(["Name","Found","Confidence"])
    parametersFormLayout.addRow(self.objectTable)

    #
    # Adjust Confidence Thresholds
    #
    confidenceThresholdsCollapsibleButton = ctk.ctkCollapsibleButton()
    confidenceThresholdsCollapsibleButton.text = "Confidence Thresholds"
    self.layout.addWidget(confidenceThresholdsCollapsibleButton)

    confidenceFormLayout = qt.QFormLayout(confidenceThresholdsCollapsibleButton)

    self.confidenceSlider = qt.QSlider(0x1) #horizontal slider
    self.confidenceSlider.setRange(0,100)
    self.confidenceSlider.setTickInterval(5)
    self.confidenceSlider.setTickPosition(2) #Ticks appear below slider
    self.confidenceSlider.setSliderPosition(80)
    self.confidenceSlider.setToolTip("Set the minimum degree of confidence that must be met for an object to be considered found")
    confidenceFormLayout.addRow("Confidence: ",self.confidenceSlider)
    self.confidenceLabel = qt.QLabel("80%")
    confidenceFormLayout.addRow(self.confidenceLabel)

    self.recordingPlayWidget = slicer.qMRMLSequenceBrowserPlayWidget()
    parametersFormLayout.addRow(self.recordingPlayWidget)

    self.recordingSeekWidget = slicer.qMRMLSequenceBrowserSeekWidget()
    parametersFormLayout.addRow(self.recordingSeekWidget)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    #self.classifyAllFramesButton.connect('clicked(bool)', self.onClassifyAllFramesClicked)
    self.modelSelector.connect('currentIndexChanged(int)',self.onModelSelected)
    self.confidenceSlider.connect('sliderMoved(int)',self.onConfidenceChanged)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
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

  def selectRecordingNode(self):
    sequenceNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLSequenceBrowserNode')
    selectedNode = sequenceNodes.GetItemAsObject(0)
    self.recordingPlayWidget.setMRMLSequenceBrowserNode(selectedNode)
    self.recordingSeekWidget.setMRMLSequenceBrowserNode(selectedNode)

  def createWebcamPlusConnector(self):
    try:
        webcamConnectorNode = slicer.util.getNode('WebcamPlusConnector')
    except slicer.util.MRMLNodeNotFoundException:
    #if not webcamConnectorNode:
      webcamConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      #webcamConnectorNode.SetLogErrorIfServerConnectionFailed(False)
      webcamConnectorNode.SetName('WebcamPlusConnector')
      slicer.mrmlScene.AddNode(webcamConnectorNode)
      # hostNamePort = self.parameterNode.GetParameter('PlusWebcamServerHostNamePort')
      hostNamePort = "localhost:18944"
      [hostName, port] = hostNamePort.split(':')
      webcamConnectorNode.SetTypeClient(hostName, int(port))
      logging.debug('Webcam PlusConnector Created')
    return webcamConnectorNode
  '''
  def createClassifierConnector(self):
    try:
        classifierConnectorNode = slicer.util.getNode('ClassifierPlusConnector')
    except slicer.util.MRMLNodeNotFoundException:
    #if not webcamConnectorNode:
      classifierConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
      #webcamConnectorNode.SetLogErrorIfServerConnectionFailed(False)
      classifierConnectorNode.SetName('ClassifierPlusConnector')
      slicer.mrmlScene.AddNode(classifierConnectorNode)
      # hostNamePort = self.parameterNode.GetParameter('PlusWebcamServerHostNamePort')
      hostNamePort = "localhost:18946"
      [hostName, port] = hostNamePort.split(':')
      classifierConnectorNode.SetTypeServer(int(port))
      logging.debug('Webcam Classifier Connector Created')
    return classifierConnectorNode

  def createLabelConnector(self):
      try:
          labelConnectorNode = slicer.util.getNode('LabelClassifierPlusConnector')
      except slicer.util.MRMLNodeNotFoundException:
          # if not webcamConnectorNode:
          labelConnectorNode = slicer.vtkMRMLIGTLConnectorNode()
          # webcamConnectorNode.SetLogErrorIfServerConnectionFailed(False)
          labelConnectorNode.SetName('LabelClassifierPlusConnector')
          slicer.mrmlScene.AddNode(labelConnectorNode)
          # hostNamePort = self.parameterNode.GetParameter('PlusWebcamServerHostNamePort')
          hostNamePort = "localhost:18947"
          [hostName, port] = hostNamePort.split(':')
          labelConnectorNode.SetTypeClient(hostName, int(port))
          logging.debug('Label Classifier Connector Created')
      return labelConnectorNode
  '''


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
    self.applyButton.enabled = False

  def onApplyButton(self):
    if self.applyButton.text == "Start":
      self.logic.run(self.objectTable,self.confidenceSlider,self.modelSelector.currentText)
      self.applyButton.setText("Stop")
    else:
      self.logic.stopClassifier()
      self.applyButton.setText("Start")

  def onClassifyAllFramesClicked(self):
      self.selectRecordingNode()
      self.logic.classifyAllFrames(self.modelSelector.currentText)

  def onModelSelected(self):
    if self.modelSelector.currentText != "Select model":
      self.applyButton.enabled = True
      self.currentModelDirectory = os.path.join(self.moduleDir, os.pardir, "Models", self.modelSelector.currentText)
      modelObjectClasses = os.listdir(os.path.join(self.currentModelDirectory,"training_photos"))
      self.currentObjectClasses = [dir for dir in modelObjectClasses if dir.find(".") == -1]
      self.objectTable.setRowCount(len(self.currentObjectClasses))
      for i in range (len(self.currentObjectClasses)):
        self.objectTable.setItem(i,0,qt.QTableWidgetItem(self.currentObjectClasses[i]))
        self.objectTable.setItem(i,1,qt.QTableWidgetItem("No"))

  def onConfidenceChanged(self):
    self.confidenceLabel.text = str(self.confidenceSlider.sliderPosition) + "%"

#
# CNN_Image_ClassifierLogic
#

class CNN_Image_ClassifierLogic(ScriptedLoadableModuleLogic):

  def run(self,objectTable,confidenceSlider,modelName):
    self.moduleDir = os.path.dirname(slicer.modules.collect_training_images.path)
    self.currentModelDirectory = os.path.join(self.moduleDir, os.pardir, "Models", modelName)
    modelObjectClasses = os.listdir(os.path.join(self.currentModelDirectory, "training_photos"))
    self.currentObjectClasses = [dir for dir in modelObjectClasses if dir.find(".") == -1]
    self.numObjects = len(self.currentObjectClasses)

    self.lastClassifiedTime = 0
    self.stopClassifierClicked = False
    self.modelName = modelName
    self.runWithWidget = True
    self.objectTable = objectTable
    self.numObjects = self.objectTable.rowCount
    self.confidenceSlider = confidenceSlider
    self.webcamReference = slicer.util.getNode('Webcam_Reference')

    model_file = 'c:/Users/hisey/Documents/ImageClassificationRepo/Models/' + self.modelName + '/trained_model/output_graph.pb'
    self.label_file = 'c:/Users/hisey/Documents/ImageClassificationRepo/Models/' + self.modelName + '/trained_model/output_labels.txt'
    input_layer = 'Placeholder'
    output_layer = 'final_result'

    self.graph = self.load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    self.input_operation = self.graph.get_operation_by_name(input_name)
    self.output_operation = self.graph.get_operation_by_name(output_name)
    self.currentLabel = ""
    self.lastUpdateSec = 0
    self.imageDataModifiedObserver = self.webcamReference.AddObserver(
            slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, self.onWebcamImageModified)

  def onWebcamImageModified(self, caller, eventid):
      if not time.time() - self.lastClassifiedTime < 0.5:
        imdata = self.getVtkImageDataAsOpenCVMat('Webcam_Reference')
        (self.label, self.percentage) = self.classify_image(self.numObjects, self.graph, imdata, self.label_file, self.input_operation,
                                  self.output_operation)
        self.getFoundObject()
        self.lastClassifiedTime = time.time()

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

  def load_graph(self, model_file):
      graph = tf.Graph()
      graph_def = tf.GraphDef()

      with open(model_file, "rb") as f:
          graph_def.ParseFromString(f.read())
      with graph.as_default():
          tf.import_graph_def(graph_def)

      return graph

  def read_tensor_from_image_file(self, image,
                                  input_height=244,
                                  input_width=244,
                                  input_mean=0,
                                  input_std=244):

      float_caster = tf.cast(image, tf.float32)  # instead of using file reader, pass in image as 3D numpy array
      dims_expander = tf.expand_dims(float_caster, 0)
      resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
      normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
      sess = tf.Session()
      result = sess.run(normalized)

      return result

  def load_labels(self, label_file):
      label = []
      proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
      for l in proto_as_ascii_lines:
          label.append(l.rstrip())
      return label

  def classify_image(self,numClasses, graph, file_name, label_file, input_operation, output_operation):
      input_height = 224
      input_width = 224
      input_mean = 0
      input_std = 224

      t = self.read_tensor_from_image_file(
          file_name,
          input_height=input_height,
          input_width=input_width,
          input_mean=input_mean,
          input_std=input_std)

      with tf.Session(graph=graph) as sess:
          results = sess.run(output_operation.outputs[0], {
              input_operation.outputs[0]: t
          })
      results = np.squeeze(results)

      top_k = results.argsort()[-numClasses:][::-1]
      labels = self.load_labels(label_file)
      topLabel = labels[0:numClasses]
      topResults = np.array(results[0:numClasses])
      topScore = topResults[0]
      topScoreInd = 0
      for i in range(numClasses):
          if topResults[i] > topScore:
              topScore = topResults[i]
              topScoreInd = i

      return (topLabel[topScoreInd], topResults)

  def runWithoutWidget(self,modelName):
    self.modelName = modelName
    self.runWithWidget = False
    self.currentLabel = ""
    self.confidences = []
    self.lastUpdateSec = 0
    #self.webcamReference = slicer.util.getNode('Webcam_Reference')
    try:
        self.labelNode = slicer.util.getNode('labelConnector')
    except slicer.util.MRMLNodeNotFoundException:
        self.labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode","labelConnector")
    self.labelObserver = self.labelNode.AddObserver(slicer.vtkMRMLTextNode.TextModifiedEvent, self.onLabelModified)
    self.moduleDir = os.path.dirname(slicer.modules.collect_training_images.path)
    self.currentModelDirectory = os.path.join(self.moduleDir, os.pardir, "Models",modelName)
    modelObjectClasses = os.listdir(os.path.join(self.currentModelDirectory, "training_photos"))
    self.currentObjectClasses = [dir for dir in modelObjectClasses if dir.find(".") == -1]
    self.numObjects = len(self.currentObjectClasses)

  def onLabelModified(self, caller, eventID):
      [self.currentLabel, self.confidences, self.currentConfidence] = self.getFoundObject()

  def classifyAllFrames(self,modelName):
    self.runWithoutWidget(modelName)
    playWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserPlayWidget")
    playWidgetButtons = playWidget[0].findChildren('QPushButton')
    playWidgetButtons[0].click() #Move to first frame
    seekWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserSeekWidget")
    seekWidget = seekWidget[0]
    self.totalNumFrames = seekWidget.slider_IndexValue.maximum
    for i in range (self.totalNumFrames):
        #self.classifyImage()
        confidenceAndLabel = list(self.confidences)
        if self.currentLabel == 'anesthetic\n':
            confidenceAndLabel.append(1)
        elif self.currentLabel == 'syringe\n':
            confidenceAndLabel.append(2)
        elif self.currentLabel == 'guidewire_casing\n':
            confidenceAndLabel.append(3)
        elif self.currentLabel == 'scalpel\n':
            confidenceAndLabel.append(4)
        elif self.currentLabel == 'dilator\n':
            confidenceAndLabel.append(5)
        elif self.currentLabel == 'catheter\n':
            confidenceAndLabel.append(6)
        elif self.currentLabel == 'guidewire\n':
            confidenceAndLabel.append(7)
        else:
            confidenceAndLabel.append(0)
        confidenceAndLabel.append(self.currentLabel)
        self.writeToCSV(confidenceAndLabel)
        self.moveToNextFrame()
    self.stopClassifier()

  def moveToNextFrame(self):
    playWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserPlayWidget")
    buttons = playWidget[0].findChildren('QPushButton')
    buttons[3].click()

  def writeToCSV(self, frameLabels):
      import csv
      #mode = 'wb'
      FileName = "LabeledFrames" + os.extsep + "csv"
      labeledFramesFilePath = os.path.join(self.moduleDir, FileName)
      with open(labeledFramesFilePath, 'ab') as csvfile:
          fileWriter = csv.writer(csvfile)
          #for frame in range(0, len(frameLabels)):
          fileWriter.writerow(frameLabels)

  def stopClassifier(self):
    self.stopClassifierClicked = True
    self.webcamReference.RemoveObserver(self.imageDataModifiedObserver)
    self.imageDataModifiedObserver = None

  def updateObjectTable(self):
    for i in range(self.numObjects):
      if self.currentLabel == str(self.objectTable.item(i,0).text()) and float(self.confidences[i]) > self.confidenceSlider.value/100.0:
        self.objectTable.setItem(i,1,qt.QTableWidgetItem("Yes"))
        self.objectTable.setItem(i,2,qt.QTableWidgetItem(str(round(self.confidences[i]*100,2))))
        self.currentConfidence = self.confidences[i]
      else:
        self.objectTable.setItem(i, 1, qt.QTableWidgetItem("No"))
        self.objectTable.setItem(i, 2, qt.QTableWidgetItem(str(round(self.confidences[i]*100,2))))

  def getFoundObject(self):
    imgClass = 0
    if self.label != None:
        self.currentLabel = self.label
        self.confidences = self.percentage
        if self.runWithWidget == True:
            self.updateObjectTable()
    else:
        self.currentLabel = "no label"
        self.currentConfidence = [1]
    while imgClass < self.numObjects and self.currentLabel != self.currentObjectClasses[imgClass]:
      imgClass += 1
    if imgClass < self.numObjects and self.currentLabel == self.currentObjectClasses[imgClass]:
      self.currentConfidence = self.confidences[imgClass]
    else:
        self.currentConfidence = self.confidences[4]
    return [self.currentLabel, self.confidences, self.currentConfidence]



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

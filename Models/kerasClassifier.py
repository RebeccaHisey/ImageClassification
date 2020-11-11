import os
import time
import cv2
import sys
import numpy
import random
import argparse
import SimpleITK as sitk
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from pyIGTLink import pyIGTLink

def loadCNNModel(modelFolder,modelName):
    structureFileName = modelName +'.json'
    weightsFileName = modelName+'.h5'
    with open(os.path.join(modelFolder,structureFileName),"r") as modelStructureFile:
        JSONModel = modelStructureFile.read()
    model = model_from_json(JSONModel)
    model.load_weights(os.path.join(modelFolder,weightsFileName))
    adam = tensorflow.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def loadLSTMModel(modelFolder, modelName):
    structureFileName = modelName + '.json'
    weightsFileName = modelName + '.h5'
    with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
        JSONModel = modelStructureFile.read()
    model = model_from_json(JSONModel)
    model.load_weights(os.path.join(modelFolder, weightsFileName))
    adam = tensorflow.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def classifyImage(image,prevSequence,cnnModel,lstmModel):

    labels = ['anesthetic','dilator','insert_catheter','insert_guidewire','insert_needle','nothing','remove_guidewire','scalpel']
    toolLabels = ['anesthetic','catheter','dilator','guidewire','guidewire_casing','nothing','scalpel','syringe']
    resized = cv2.resize(image, (224, 224))
    resized = numpy.expand_dims(resized,axis=0)
    toolClassification = cnnModel.predict(numpy.array(resized))
    toolLabelIndex = numpy.argmax(toolClassification)
    toolLabel = toolLabels[toolLabelIndex]
    print (str(toolLabel) + str(toolClassification))
    newSequence = numpy.append(prevSequence[1:],toolClassification,axis=0)
    taskClassification = lstmModel.predict(numpy.array([newSequence]))
    labelIndex = numpy.argmax(taskClassification)
    label = labels[labelIndex]
    return(label,taskClassification,newSequence)
    #return(toolLabel,toolClassification,newSequence)

def convertTooltoTaskLabel(toolClassification):
    taskClassification = numpy.zeros(8)
    print(toolClassification)
    taskClassification[0] = toolClassification[0][0]
    taskClassification[1] = toolClassification[0][2]
    taskClassification[2] = toolClassification[0][1]
    taskClassification[3] = toolClassification[0][4]
    taskClassification[4] = toolClassification[0][7]
    taskClassification[5] = toolClassification[0][5]
    taskClassification[6] = toolClassification[0][3]
    taskClassification[7] = toolClassification[0][6]
    return taskClassification

def CNNClassifyImage(image,cnnModel):
    labels = ['anesthetic','dilator','insert_catheter','insert_guidewire','insert_needle','nothing','remove_guidewire','scalpel']
    resized = cv2.resize(image, (224, 224))
    resized = numpy.expand_dims(resized,axis=0)
    toolClassification = cnnModel.predict(numpy.array(resized))
    taskClassification = convertTooltoTaskLabel(toolClassification)
    tasklLabelIndex = numpy.argmax(taskClassification)
    taskLabel = labels[tasklLabelIndex]
    print (str(taskLabel) + str(taskClassification))
    return(taskLabel,taskClassification)

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    cnn_model_name = FLAGS.cnn_model_name
    lstm_model_name = FLAGS.lstm_model_name
    modelFolder =FLAGS.model_directory
    cnnModel = loadCNNModel(modelFolder,cnn_model_name)
    lstmModel = loadLSTMModel(modelFolder,lstm_model_name)
    prevSequence = numpy.zeros((50,8))

    print("Server starting...")
    server = pyIGTLink.PyIGTLinkServer(port=18947, localServer=True)
    server.start()
    print("Server running...")

    print("Client starting...")
    client = pyIGTLink.PyIGTLinkClient(host="127.0.0.1", port=18946)
    client.start()
    print("Client running...")
    lastMessageTime = time.time()
    ImageReceived = False
    frameCount = 0
    try:
        while (not ImageReceived) or (ImageReceived and time.time() - lastMessageTime < 15):
            messages = client.get_latest_messages()
            if len(messages) > 0:
                for message in messages:
                    if message._type == "IMAGE":
                        frameCount +=1
                        ImageReceived = True
                        lastMessageTime = time.time()
                        image = message._image
                        image = image[0]
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        print(time.time())
                        (label, confidence,imgSequence) = classifyImage(image,prevSequence,cnnModel,lstmModel)

                        print(time.time())
                        labelMessage = pyIGTLink.StringMessage(str(label)+ "," + str(confidence), device_name='labelConnector')

                        server.send_message(labelMessage)
                        print (str(label) + str(confidence))
                        print(frameCount)
                        prevSequence = imgSequence
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--cnn_model_name',
      type=str,
      default='',
      help='Name of cnn model.'
  )
  parser.add_argument(
      '--lstm_model_name',
      type=str,
      default='',
      help='Name of lstm model.'
  )
  parser.add_argument(
      '--model_directory',
      type=str,
      default='',
      help='Location of model.'
  )
FLAGS, unparsed = parser.parse_known_args()
main()
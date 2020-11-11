from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import sys
import os
import subprocess
import itertools
import time
import tensorflow as tf
import logging
from pyIGTLink import pyIGTLink

FLAGS = None

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(image,
                                input_height=224,
                                input_width=224,
                                input_mean=0,
                                input_std=224):

  '''
  input_name = "file_reader"
  output_name = "normalized"

  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(
          file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(
          tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
      image_reader = tf.image.decode_jpeg(
          file_reader, channels=3, name="jpeg_reader")
  '''

  #image_reader = file_name
  float_caster = tf.cast(image, tf.float32) #instead of using file reader, pass in image as 3D numpy array
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def classify_image(numClasses,graph,file_name,label_file,input_operation,output_operation):
    input_height = 224
    input_width = 224
    input_mean = 0
    input_std = 224

    t = read_tensor_from_image_file(
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
    labels = load_labels(label_file)
    topLabel = labels[0:numClasses]
    topResults = np.array(results[0:numClasses])
    topScore = topResults[0]
    topScoreInd = 0
    for i in range(numClasses):
        if topResults[i] > topScore:
            topScore = topResults[i]
            topScoreInd = i

    return (topLabel[topScoreInd],topResults)

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model_name = FLAGS.model_name
    #model_file = 'trained_model/'+os.environ['MODELNAME']+'/output_graph.pb'
    model_file = 'c:/Users/hisey/Documents/ImageClassificationRepo/Models/'+ model_name + '/trained_model/output_graph.pb'
    #label_file = 'trained_model/'+os.environ['MODELNAME']+'/output_labels.txt'
    label_file = 'c:/Users/hisey/Documents/ImageClassificationRepo/Models/'+ model_name + '/trained_model/output_labels.txt'
    input_layer = 'Placeholder'
    output_layer = 'final_result'
    #numClasses = int(os.environ['NUMCLASSES'])
    numClasses = int(8)

    graph = load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    currentTime = time.time()

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
    try:
        while (not ImageReceived) or (ImageReceived and time.time() - lastMessageTime < 15):
            messages = client.get_latest_messages()
            if len(messages) > 0:
                for message in messages:
                    if message._type == "IMAGE":
                        ImageReceived = True
                        lastMessageTime = time.time()
                        image = message._image
                        image = image[0]
                        print(time.time())
                        (label, p) = classify_image(numClasses, graph, image, label_file, input_operation,
                                                    output_operation)
                        print(time.time())
                        labelMessage = pyIGTLink.StringMessage(str(label)+ "," + str(p), device_name='labelConnector')

                        server.send_message(labelMessage)
                        print (str(label) + str(p))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

#main()
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_name',
      type=str,
      default='',
      help='Name of model.'
  )
FLAGS, unparsed = parser.parse_known_args()
main()
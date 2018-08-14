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
import label_image as labIm
import logging

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  image_reader = file_name
  float_caster = tf.cast(image_reader, tf.float32) #instead of using file reader, pass in image as 3D numpy array
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

def classify_image(graph,file_name,label_file,input_operation,output_operation):
	input_height = 299
	input_width = 299
	input_mean = 0
	input_std = 255

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

	top_k = results.argsort()[-5:][::-1]
	labels = load_labels(label_file)
	topLabel = labels[0:6]
	topResults = np.array(results[0:6])
	topScore = topResults[0]
	topScoreInd = 0
	for i in range(2):
		if topResults[i] > topScore:
			topScore = topResults[i]
			topScoreInd = i

	return (topLabel[topScoreInd],topResults)

def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	model_file = 'trained_model/output_graph.pb'
	label_file = 'trained_model/output_labels.txt'
	input_layer = 'Placeholder'
	output_layer = 'final_result'

	graph = load_graph(model_file)
	input_name = "import/" + input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name)
	output_operation = graph.get_operation_by_name(output_name)
	currentTime = time.time()

	filemodTime1 = 0

	while True:
		filemodTime2 = os.path.getmtime('textLabels.txt')
		startTime = time.time()
		while filemodTime1 == filemodTime2 and time.time() - startTime < 5:
			time.sleep(0.2)
			filemodTime2 = os.path.getmtime('textLabels.txt')
		imgSize = np.load('pictureSize.npy',fix_imports=True)
		imageFile = np.load('picture.npy',fix_imports=True)
		imageFile.reshape(imgSize)

		filemodTime1 = time.time()

		(label, p) = classify_image(graph,imageFile,label_file,input_operation,output_operation)
		with open('textLabels.txt','w') as f:
			f.write(label + "\n" + str(p[0]) + '\n' + str(p[1]))
		currentTime = time.time()



main()
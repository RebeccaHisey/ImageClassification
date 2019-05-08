from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import sys
import os
import subprocess
import time
import tensorflow as tf
import logging

def main():
	trainingPhotoDirectory = "--image_dir=./" + os.environ['MODELNAME'] + "/training_photos"
	outputGraph = "--output_graph=./" + os.environ['MODELNAME'] + "/trained_model/output_graph.pb"
	outputLabels = "--output_labels=./" + os.environ['MODELNAME'] + "/trained_model/output_labels.txt"
	summariesDirectory = "--summaries_dir=./" + os.environ['MODELNAME'] + "/trained_model/retrain_logs"
	bottleneckDirectory = "--bottleneck_dir=./" + os.environ['MODELNAME'] + "/trained_model/bottleneck"
	savedModelDirectory = "--saved_model_dir=./" + os.environ['MODELNAME'] + "/trained_model"
	numTrainingSteps = "--how_many_training_steps="+os.environ['NUMTRAININGSTEPS']
	batchSize = "--train_batch_size="+os.environ['TRAINBATCHSIZE']

	cmd = ["python","retrain.py",numTrainingSteps,batchSize,trainingPhotoDirectory,outputGraph,outputLabels,summariesDirectory,bottleneckDirectory,savedModelDirectory]
	p = subprocess.Popen(cmd,stdin = subprocess.PIPE, shell=False)
	p.communicate()
	#while p.poll() is None:
	#	l = p.stdout.readline()  # This blocks until it receives a newline.
	#	logging.info(l)
	# When the subprocess terminates there might be unconsumed output
	# that still needs to be processed.
	#logging.info(p.stdout.read())
	#(output, error) = p.communicate()
	cmd = ["tensorboard","--logdir=/WatchFinder/trained_model/retrain_logs"]
	q = subprocess.call(cmd,stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True)
	#(output1,error1) = q.communicate()
	#print (output)
	#print (error)
	#print(output1)
	#print(error1)
main()
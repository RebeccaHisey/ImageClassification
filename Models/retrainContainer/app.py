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
	cmd = ["python","retrain.py","--image_dir=./WatchFinder/training_photos","--output_graph","./WatchFinder/trained_model/output_graph.pb","--output_labels",
		   "./WatchFinder/trained_model/output_labels.txt","--summaries_dir","./WatchFinder/trained_model/retrain_logs","--bottleneck_dir","./WatchFinder/trained_model/bottleneck","--saved_model_dir",
		   "./WatchFinder/trained_model"]
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
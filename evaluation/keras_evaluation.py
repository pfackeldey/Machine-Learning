#!/usr/bin/env python

import ROOT
# disable ROOT internal argument parser
ROOT.PyConfig.IgnoreCommandLineOptions = True

import argparse
from array import array
import yaml
import pickle
import numpy as np
import os
import sys

from keras.models import load_model

base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)

from utils.treetools import *

def keras_evaluation():
	parser = argparse.ArgumentParser(description="Application of keras model.",
		                             fromfile_prefix_chars="@", conflict_handler="resolve")
	parser.add_argument("config", help = "Path to config")
	parser.add_argument("--files", nargs='+', help = "Path to ROOT file.")
	parser.add_argument("--tree", default = "latino", help="Name of the tree.")

	args = parser.parse_args()

	config = yaml.load(open(args.config, "r"))

	# Load keras model and preprocessing
	classifiers = []
	preprocessing = []
	for c, p in zip(config["classifiers"], config["preprocessing"]):
		classifiers.append(load_model(c))
		preprocessing.append(pickle.load(open(p, "rb")))

	print "Currently processing {}".format(args.file)
	files = args.files

	for file in files:
		path = file + "/" +  args.tree

		with TreeExtender(path) as extender:

			values = []
			for feature in config["features"]:
				values.append(array("f", [-999]))
				extender.tree.SetBranchAddress(feature, values[-1])

			event_branch = config["event_branch"]

			extender.addBranch("ml_max_score", nLeaves = 1, unpackBranches = None)
			extender.addBranch("ml_max_index", nLeaves = 1, unpackBranches = None)

			for entry in extender:

				# Get event number and calculate method's response
				event = int(getattr(extender.tree, event_branch))

				values_stacked = np.hstack(values).reshape(1, len(values))

				# preprocessing
				values_preprocessed = preprocessing[event % 2].transform(values_stacked)
				response = classifiers[event % 2].predict(values_preprocessed)
				response = np.squeeze(response)

				# Find max score and index
				for i, r in enumerate(response):
					if r > entry.ml_max_score[0]:
						entry.ml_max_score[0] = r
						entry.ml_max_index[0] = i

if __name__ == "__main__":
    keras_evaluation()

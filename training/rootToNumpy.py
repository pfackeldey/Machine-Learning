#!/usr/bin/env python

import ROOT
# disable ROOT internal argument parser
ROOT.PyConfig.IgnoreCommandLineOptions = True
import root_numpy
import numpy as np
np.random.seed(1234)

import argparse
import yaml
import os
import sys

def rootToNumpy(args_from_script=None):

    parser = argparse.ArgumentParser(description="Dump trees to numpy arrays.",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")
    parser.add_argument("config", help="Path to training config")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"))

    for fold in range(2):

	    features = config["features"]

	    filename = config["trainingssets"][fold]

	    x = []
	    y = []
	    w = []
	    rfile = ROOT.TFile(filename, "READ")
	    classes = config["classes"]
	    for i_class, class_ in enumerate(classes):
		tree = rfile.Get(class_)
		if tree == None:
		    print "Tree %s not found in file %s.", class_, filename
		    raise Exception

		# Get inputs for this class
		x_class = np.zeros((tree.GetEntries(), len(features)))
		x_conv = root_numpy.tree2array(tree, branches=features)
		for i_feature, feature in enumerate(features):
		    x_class[:, i_feature] = x_conv[feature]
		x.append(x_class)

		# Get weights
		w_class = np.zeros((tree.GetEntries(), 1))
		w_conv = root_numpy.tree2array(
		    tree, branches=[config["event_weights"]])
		w_class[:, 0] = w_conv[config["event_weights"]] * config[
		    "class_weights"][class_]
		w.append(w_class)

		# Get targets for this class
		y_class = np.zeros((tree.GetEntries(), len(classes)))
		y_class[:, i_class] = np.ones((tree.GetEntries()))
		y.append(y_class)

	    # Stack inputs, targets and weights to a Keras-readable dataset
	    x = np.vstack(x)  # inputs
	    y = np.vstack(y)  # targets
	    w = np.vstack(w) * config["global_weight"]  # weights
	    w = np.squeeze(w)  # needed to get weights into keras

	    folder = '~/Machine-Learning/arrays/'
	    if not os.path.exists(folder):
		os.makedirs(folder)

	    # Dump numpy arrays
	    np.save(folder+'x_fold{}.npy'.format(fold), x)
	    np.save(folder+'y_fold{}.npy'.format(fold), y)
	    np.save(folder+'weights_fold{}.npy'.format(fold), w)

if __name__ == "__main__" and len(sys.argv) > 1:
    rootToNumpy()

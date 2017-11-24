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

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
from matplotlib import cm

from keras.models import load_model

parser = argparse.ArgumentParser(description="Perform multiclassification NN training with kPyKeras (TMVA).",
                                 fromfile_prefix_chars="@", conflict_handler="resolve")
parser.add_argument("model", help="Path to modelfile (XY.h5)")
parser.add_argument("config", help="Path to training config")

args = parser.parse_args()


def confusionplot(confusion, classes, output):
    plt.figure(figsize=(2.5 * confusion.shape[0], 2.0 * confusion.shape[1]))
    axis = plt.gca()
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            axis.text(
                i + 0.5,
                j + 0.5,
                markup.format(confusion[-1 - j, i]),
                ha='center',
                va='center')
    q = plt.pcolormesh(confusion[::-1], cmap='Wistia')
    cbar = plt.colorbar(q)
    cbar.set_label("Sum of event weights", rotation=270, labelpad=50)
    plt.xticks(
        np.array(range(len(classes))) + 0.5, classes, rotation='vertical')
    plt.yticks(
        np.array(range(len(classes))) + 0.5,
        classes[::-1],
        rotation='horizontal')
    plt.xlim(0, len(classes))
    plt.ylim(0, len(classes))
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(filename, bbox_inches='tight')


# load keras model
model = load_model(args.model)

# load yaml training config
config = yaml.load(open(args.config, "r"))

# create empty confusion matrix
confusion = np.zeros(
    (len(config_train["classes"]), len(config_train["classes"])),
    dtype=np.float)

# WORK IN PROGRESS

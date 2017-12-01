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

from sklearn import model_selection

base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)

from utils.model import KerasModels

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def multiclassNeuralNetwork(args_from_script=None):

    parser = argparse.ArgumentParser(description="Perform multiclassification NN training with Keras.",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")
    parser.add_argument("--fold", default=0, choices=[0, 1],
                        help="Training fold. [Default: %(default)s]")
    parser.add_argument("--epochs", default=150,
                        help="Number of training epochs. [Default: %(default)s]")
    parser.add_argument("--learning-rate", default=0.00005,
                        help="Learning rate of NN. [Default: %(default)s]")
    parser.add_argument("--batch-size", default=7500,
                        help="Batch size for training. [Default: %(default)s]")
    parser.add_argument("--early-stopping", default=False, action='store_true',
                        help="Stop training if loss increases again. [Default: %(default)s]")
    parser.add_argument("config", help="Path to training config")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"))

    features = config["features"]

    filename = config["trainingssets"][args.fold]

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

    # Split data in training and testing
    x_train, x_test, y_train, y_test, w_train, w_test = model_selection.train_test_split(
        x, y, w, test_size=1.0 - config["train_test_split"], random_state=1234)

    # Add callbacks
    callbacks = []
    callbacks.append(TensorBoard(log_dir='./logs',
                                 histogram_freq=1, write_graph=True, write_images=True))
    callbacks.append(
        ModelCheckpoint(filepath="fold{}_multiclass_model.h5".format(args.fold), save_best_only=True, verbose=1))
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=2,
                                       verbose=0, mode='auto'))

    model = KerasModels(n_features=len(config["features"]), n_classes=len(
        config["classes"]), learning_rate=args.learning_rate, plot_model=False, modelname="multiclass_model_fold{}.h5".format(args.fold))
    model.multiclass_MSSM_HWW_model()

    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        validation_data=(x_test, y_test, w_test),
        batch_size=args.batch_size,
        nb_epoch=args.epochs,
        shuffle=True,
        callbacks=callbacks)


if __name__ == "__main__" and len(sys.argv) > 1:
    try:
        import tensorflow as tf
        tf.python.control_flow_ops = tf
        multiclassNeuralNetwork()
    except AttributeError:
        multiclassNeuralNetwork()

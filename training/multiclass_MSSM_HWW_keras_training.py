#!/usr/bin/env python

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
import utils.confusionmatrix

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

    # load trainings data and weights
    x = np.load('x.npy')
    y = np.load('y.npy')
    w = np.load('weights.npy')

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
    multiclass_model = getattr(model, "multiclass_MSSM_HWW_model")

    multiclass_model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        validation_data=(x_test, y_test, w_test),
        batch_size=args.batch_size,
        nb_epoch=args.epochs,
        shuffle=True,
        callbacks=callbacks)

    # plot loss
    f = plt.figure()
    plt.plot(fit.history["loss"])
    plt.plot(fit.history["val_loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["training loss", "validation loss"], loc="best")
    f.savefig("loss.png")

    # plot accuracy
    f = plt.figure()
    plt.plot(fit.history["acc"])
    plt.plot(fit.history["val_acc"])
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(["training accuracy", "validation accuracy"], loc="best")
    f.savefig("accuracy.png")

    # testing
    [loss, accuracy] = model.evaluate(x_test, y_test, verbose=0)

    val_loss = fit.history["val_loss"][-1]
    val_accuracy = fit.history["val_acc"][-1]

    # predicted probabilities for the test set
    Yp = model.predict(x_test)
    yp = np.argmax(Yp, axis=1)

    folder = 'results/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # plot confusion matrix
    plot_confusion(yp, y_test, config["classes"],
                   fname=folder + 'confusion.png')


if __name__ == "__main__" and len(sys.argv) > 1:
    try:
        import tensorflow as tf
        tf.python.control_flow_ops = tf
        multiclassNeuralNetwork()
    except AttributeError:
        multiclassNeuralNetwork()

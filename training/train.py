#!/usr/bin/env python

import numpy as np
np.random.seed(1234)

from collections import Counter
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
    parser.add_argument("--epochs", default=20,
                        help="Number of training epochs. [Default: %(default)s]")
    parser.add_argument("--learning-rate", default=0.0001,
                        help="Learning rate of NN. [Default: %(default)s]")
    parser.add_argument("--batch-size", default=10000,
                        help="Batch size for training. [Default: %(default)s]")
    parser.add_argument("--early-stopping", default=False, action='store_true',
                        help="Stop training if loss increases again. [Default: %(default)s]")
    parser.add_argument("config", help="Path to training config")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"))

    folder = '/home/mf278754/master/arrays/'

    # load trainings data and weights
    x = np.load(folder + 'x_fold{}.npy'.format(args.fold))
    y = np.load(folder + 'y_fold{}.npy'.format(args.fold))
    w = np.load(folder + 'weights_fold{}.npy'.format(args.fold))
    w = w * config["global_weight"]

    # Split data in training and testing
    x_train, x_test, y_train, y_test, w_train, w_test = model_selection.train_test_split(
        x, y, w, test_size=1.0 - config["train_test_split"], random_state=1234)

    def get_class_weights(y):
        counter = Counter(y)
        majority = 1.  # max(counter.values())
        return {cls: float(majority / count) for cls, count in counter.items()}

    # Add callbacks
    callbacks = []
    # callbacks.append(TensorBoard(log_dir='/home/mf278754/master/logs',
    #                             histogram_freq=1, write_graph=True, write_images=True))
    callbacks.append(
        ModelCheckpoint(filepath="/home/mf278754/master/fold{}_multiclass_model.h5".format(args.fold), save_best_only=True, verbose=1))
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=2,
                                       verbose=0, mode='auto'))

    # preprocessing
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    scaler = preprocessing.StandardScaler().fit(x_test)
    x_test_scaled = scaler.transform(x_test)

    model = KerasModels(n_features=len(config["features"]), n_classes=len(
        config["classes"]), learning_rate=args.learning_rate, plot_model=False, modelname="multiclass_model_fold{}.h5".format(args.fold))
    keras_model = model.multiclass_MSSM_HWW_model()
    fit = keras_model.fit(
        x_train_scaled,
        y_train,
        sample_weight=w_train,
        validation_data=(x_test_scaled, y_test),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        callbacks=callbacks)

    folder_result = '/home/mf278754/master/results/'
    if not os.path.exists(folder_result):
        os.makedirs(folder_result)

    # dump loss and accuracy to numpy arrays
    np.save(folder_result + 'loss.npy', fit.history["loss"])
    np.save(folder_result + 'val_loss.npy', fit.history["val_loss"])
    np.save(folder_result + 'acc.npy', fit.history["acc"])
    np.save(folder_result + 'val_acc.npy', fit.history["val_acc"])


if __name__ == "__main__" and len(sys.argv) > 1:
    multiclassNeuralNetwork()

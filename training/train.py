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

from utils.model import KerasModel, Config

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


def multiclassNeuralNetwork():
    """
    Argument Parser for the training step. Mainly modifying hyperparameter.
    """
    parser = argparse.ArgumentParser(description="Perform multiclassification NN training with Keras.",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")
    parser.add_argument("--fold", default=0, choices=[0, 1],
                        help="Training fold. [Default: %(default)s]")
    parser.add_argument("--epochs", default=200,
                        help="Number of training epochs. [Default: %(default)s]")
    parser.add_argument("--learning-rate", default=0.000005,
                        help="Learning rate of NN. [Default: %(default)s]")
    parser.add_argument("--batch-size", default=1000,
                        help="Batch size for training. [Default: %(default)s]")
    parser.add_argument("--early-stopping", default=False, action='store_true',
                        help="Stop training if loss increases again. [Default: %(default)s]")
    args = parser.parse_args()

    config = Config()

    folder = base + '/NumpyConversion/'

    # load trainings data and weights
    data = np.load(folder + 'data_fold0.npz'.format(args.fold))
    x = data['x']
    y = data['y']
    w = data['w']
    w = w * config.load["global_weight"]

    # Split data in training and testing
    x_train, x_test, y_train, y_test, w_train, w_test = model_selection.train_test_split(
        x, y, w, test_size=1.0 - config.load["train_test_split"], random_state=1234)

    folder_result = base + '/results/'
    if not os.path.exists(folder_result):
        os.makedirs(folder_result)

    np.save(folder_result + 'x_fold{}_test.npy'.format(args.fold), x_test)
    np.save(folder_result + 'y_fold{}_test.npy'.format(args.fold), y_test)

    def get_class_weights(y):
        counter = Counter(y)
        majority = 1.  # max(counter.values())
        return {cls: float(majority / count) for cls, count in counter.items()}

    # Add callbacks
    callbacks = []
    # callbacks.append(TensorBoard(log_dir='/home/mf278754/master/logs',
    #                             histogram_freq=1, write_graph=True, write_images=True))
    callbacks.append(
        ModelCheckpoint(filepath=base + "/test_fold{}_multiclass_model.h5".format(args.fold), save_best_only=True, verbose=1))
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=2,
                                       verbose=0, mode='auto'))

    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5))

    # preprocessing
    import pickle
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    path_preprocessing = os.path.join(
        base, "fold{}_keras_preprocessing.pickle".format(
            args.fold))
    pickle.dump(scaler, open(path_preprocessing, 'wb'))

    # create KerasModel instance
    model = KerasModel()
    print "\033[1;33mSetting up model...\033[1;m"

    # call setter of basic model parameters
    model.lr = args.learning_rate
    model.modelname = str(args.fold)
    model.plot_model = False

    # print model parameter
    print "Number of training features is: ", model.nfeatures
    print "Number of target classes is: ", model.nclasses
    print "Learning rate is set to: ", model.lr
    print "Fully trained model name is set to: ", model.modelname
    print "Model plotting is set to: ", model.plot_model

    # setup model with new model attributes
    keras_model = model.multiclass_MSSM_HWW_model()
    print "\033[1;42mModel setup was successful!\033[1;m"

    # call keras fit function to start the training
    fit = keras_model.fit(
        x_train_scaled,
        y_train,
        sample_weight=w_train,
        validation_data=(x_test_scaled, y_test, w_test),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=2)

    # dump loss and accuracy to numpy arrays
    np.save(folder_result + 'loss.npy', fit.history["loss"])
    np.save(folder_result + 'val_loss.npy', fit.history["val_loss"])
    np.save(folder_result + 'acc.npy', fit.history["acc"])
    np.save(folder_result + 'val_acc.npy', fit.history["val_acc"])


if __name__ == "__main__":
    multiclassNeuralNetwork()

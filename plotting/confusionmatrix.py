#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import yaml

FOLD = 0

config = yaml.load(open(
    "~/Machine-Learning/config/MSSM_HWW.yaml", "r"))

# load testing data
x_test = np.load(
    '~/Machine-Learning/arrays/x_test_fold{}.npy'.format(FOLD))
y_test = np.load(
    '~/Machine-Learning/arrays/y_test_fold{}.npy'.format(FOLD))

# load model
from keras.models import load_model
model = load_model(
    '~/Machine-Learning/fold{}_multiclass_model.h5'.format(FOLD))


# preprocessing
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_test)
x_test_scaled = scaler.transform(x_test)

# predicted probabilities for the test set
Yp = model.predict(x_test_scaled, verbose=1)

# to label
yp = np.argmax(Yp, axis=-1)
y_test = np.argmax(y_test, axis=-1)


import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.viridis):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yp)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=config["classes"],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=config["classes"], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

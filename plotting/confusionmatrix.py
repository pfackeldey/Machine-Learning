#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import yaml

config = yaml.load(open(
    "/home/peter/Machine-Learning/training/configs/multiclass_MSSM_HWW_training.yaml", "r"))

# load acc and loss
#loss = np.load('loss.npy')
#val_loss = np.load('val_loss.npy')
#acc = np.load('acc.npy')
#val_acc = np.load('val_acc.npy')

FOLD = 0
# load testing data
x_test = np.load(
    '/home/peter/Machine-Learning/arrays/x_test_fold{}.npy'.format(FOLD))
y_test = np.load(
    '/home/peter/Machine-Learning/arrays/y_test_fold{}.npy'.format(FOLD))


from keras.models import load_model
model = load_model(
    '/home/peter/Machine-Learning/fold{}_multiclass_model.h5'.format(FOLD))
# testing
#[loss, accuracy] = model.evaluate(x_test, y_test, verbose=1)

#val_loss = val_loss[-1]
#val_accuracy = val_acc[-1]

# predicted probabilities for the test set
Yp = model.predict(x_test, verbose=1)

# to label
yp = np.argmax(Yp, axis=-1)
y_test = np.argmax(y_test, axis=-1)


import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
